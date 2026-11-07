import math
import os
import torch
import torch.nn as nn
from pathlib import Path
from ignite.metrics import Loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.metrics import PSNRMetric, SSIMMetric
from monai.handlers import (
    TensorBoardStatsHandler,
    TensorBoardImageHandler,
    CheckpointSaver,
    IgniteMetricHandler,
    from_engine,
)
from typing import Optional, Dict, Any, Tuple
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        network: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        loss_function: Optional[nn.Module] = None,
        max_epochs: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        log_dir: str | Path = "./log/run",
        key_val_metric: Optional[Dict[str, Any]] = None,
        additional_metrics: Optional[Dict[str, Any]] = None,
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "min",
        max_grad_norm: float = 1.0,
    ):
        self.network = network
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function or nn.L1Loss()
        self.max_epochs = max_epochs
        self.device = torch.device(device)
        self.log_dir = str(log_dir)
        self.max_grad_norm = max_grad_norm
        self.epoch = 0
        self.iteration = 0

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.patience_counter = 0
        self.best_model_path = os.path.join(self.log_dir, "best_model.pth")
        self.inference_model_path = os.path.join(
            self.log_dir, "inference_model_bundle.pt"
        )

        os.makedirs(self.log_dir, exist_ok=True)
        self.network.to(self.device)

        # 단일 SummaryWriter 생성
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.train_handlers = [
            TensorBoardStatsHandler(
                tag_name="train_loss",
                output_transform=from_engine(["loss"], first=True),
                summary_writer=self.writer,
            ),
            TensorBoardImageHandler(
                batch_transform=self._custom_batch_transform,
                output_transform=from_engine(["pred"]),
                max_channels=1,
                max_frames=4,
                epoch_level=True,
                summary_writer=self.writer,
            ),
            CheckpointSaver(
                save_dir=self.log_dir,
                save_dict={"network": self.network},
                save_interval=1,
                n_saved=3,
                save_final=True,
            ),
        ]

        self.eval_handlers = []

        use_amp = self.device.type == "cuda"
        amp_kwargs = (
            {"dtype": torch.bfloat16}
            if (use_amp and torch.cuda.is_bf16_supported())
            else None
        )

        self.trainer = SupervisedTrainer(
            device=self.device,
            max_epochs=self.max_epochs,
            network=self.network,
            train_data_loader=self.train_loader,
            optimizer=self.optimizer,
            loss_function=self.loss_function,
            train_handlers=self.train_handlers,
            prepare_batch=self._custom_prepare_batch,
            amp=use_amp,
            amp_kwargs=amp_kwargs,
        )

        @self.trainer.on("BACKWARD_COMPLETED")
        def grad_clip_handler(engine):
            if (
                getattr(engine, "amp", False)
                and getattr(engine, "scaler", None) is not None
                and self.optimizer is not None
            ):
                engine.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.network.parameters(), max_norm=self.max_grad_norm)

        self.evaluator = None
        if self.val_loader is not None:
            default_key, default_add = self._create_metrics(prefix="val")

            # 외부에서 주입된 메트릭이 있을 경우에도 안전하게 래핑 및 분리 처리
            if key_val_metric is not None:
                wrapped_metrics = {
                    k: self._ensure_ignite_metric(v) for k, v in key_val_metric.items()
                }
                if len(wrapped_metrics) > 1 and additional_metrics is None:
                    items = list(wrapped_metrics.items())
                    key_val_metric = dict([items[0]])
                    additional_metrics = dict(items[1:])
                else:
                    key_val_metric = wrapped_metrics
                    if additional_metrics is not None:
                        additional_metrics = {
                            k: self._ensure_ignite_metric(v)
                            for k, v in additional_metrics.items()
                        }
                    else:
                        # 외부에서 val_loss를 빼고 넘겼을 경우 기본 보조 지표 병합
                        additional_metrics = {
                            k: v
                            for k, v in {**default_key, **default_add}.items()
                            if k not in key_val_metric
                        }
            else:
                key_val_metric = default_key
                additional_metrics = default_add

            self.evaluator = SupervisedEvaluator(
                device=self.device,
                val_data_loader=self.val_loader,
                network=self.network,
                val_handlers=self.eval_handlers,
                key_val_metric=key_val_metric,
                additional_metrics=additional_metrics,
                prepare_batch=self._custom_prepare_batch,
                amp=use_amp,
                amp_kwargs=amp_kwargs,
            )

            @self.trainer.on("EPOCH_COMPLETED")
            def run_evaluation_and_checkpoint(engine):
                current_epoch = engine.state.epoch
                current_iteration = engine.state.iteration
                self.epoch, self.iteration = current_epoch, current_iteration

                if self.optimizer is not None:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar(
                        "learning_rate", current_lr, current_iteration
                    )

                if current_epoch % 10 == 0 or current_epoch == self.max_epochs:
                    print(f"\n[Epoch {current_epoch}] Running validation...")
                    self.evaluator.state.epoch = current_epoch
                    self.evaluator.state.iteration = current_iteration

                    val_metrics = self.eval()
                    print(f"[Epoch {current_epoch}] Validation Complete")

                    for metric_name, metric_value in val_metrics.items():
                        if isinstance(metric_value, torch.Tensor):
                            metric_value = metric_value.item()
                        self.writer.add_scalar(
                            metric_name, metric_value, current_iteration
                        )
                    self.writer.flush()

                    # 스케줄러 업데이트
                    if self.lr_scheduler is not None:
                        if isinstance(
                            self.lr_scheduler,
                            torch.optim.lr_scheduler.ReduceLROnPlateau,
                        ):
                            # val_loss가 없으면 val_psnr을 대체 사용
                            target_metric = (
                                "val_loss" if "val_loss" in val_metrics else "val_psnr"
                            )
                            val_score = val_metrics.get(target_metric, 0.0)
                            if isinstance(val_score, torch.Tensor):
                                val_score = val_score.item()
                            if math.isfinite(float(val_score)):
                                self.lr_scheduler.step(val_score)
                        else:
                            self.lr_scheduler.step()

                    # 얼리스타핑 지표 설정 (설정한 mode의 지표가 없을 경우 대비 fallback 추가)
                    if self.mode == "min":
                        current_score = val_metrics.get(
                            "val_loss", val_metrics.get("val_psnr", float("inf"))
                        )
                    else:
                        current_score = val_metrics.get(
                            "val_psnr", val_metrics.get("val_loss", float("-inf"))
                        )

                    if isinstance(current_score, torch.Tensor):
                        current_score = current_score.item()
                    current_score = float(current_score)

                    val_loss_raw = val_metrics.get("val_loss", 0.0)
                    if isinstance(val_loss_raw, torch.Tensor):
                        val_loss_raw = val_loss_raw.item()

                    # NaN/Inf 또는 0.0(발산 폴백)으로 인해 best_model.pth가 오염되는 현상 방지
                    if not math.isfinite(current_score) or not math.isfinite(
                        float(val_loss_raw)
                    ):
                        print(
                            f"[!] Non-finite validation metric detected at epoch {current_epoch} "
                            f"(score={current_score}, val_loss={val_loss_raw}). "
                            "Terminating early to protect best_model.pth!"
                        )
                        engine.terminate()
                        return

                    valid_positive = (self.mode != "max") or (current_score > 0.0)
                    is_better = valid_positive and (
                        (current_score < self.best_score - self.min_delta)
                        if self.mode == "min"
                        else (current_score > self.best_score + self.min_delta)
                    )

                    if is_better:
                        self.best_score = current_score
                        self.patience_counter = 0
                        self._save_best_model(current_epoch)
                    else:
                        self.patience_counter += 1
                        print(
                            f"[-] Validation score did not improve. Patience: {self.patience_counter}/{self.patience}"
                        )

                    if self.patience_counter >= self.patience:
                        print(f"[!] Early stopping triggered at epoch {current_epoch}!")
                        engine.terminate()

        self._setup_tqdm_bars()

    @staticmethod
    def _custom_prepare_batch(batchdata, device, non_blocking):
        if isinstance(batchdata, list):
            batchdata = batchdata[0]

        low_res = batchdata.get("quarter", batchdata.get("low_res"))
        high_res = batchdata.get("full", batchdata.get("high_res"))

        if low_res is None or high_res is None:
            raise KeyError(
                f"Batch keys not found. Available keys: {list(batchdata.keys())}"
            )

        return (
            low_res.to(device=device, non_blocking=non_blocking),
            high_res.to(device=device, non_blocking=non_blocking),
        )

    @staticmethod
    def _custom_batch_transform(batch):
        if isinstance(batch, list):
            batch = batch[0]
        low_res = batch.get("quarter", batch.get("low_res"))
        high_res = batch.get("full", batch.get("high_res"))
        return low_res, high_res

    def _ensure_ignite_metric(self, metric_obj: Any) -> Any:
        """MONAI 메트릭이 전달되었을 때 .attach() 메서드가 없다면 IgniteMetricHandler로 감싸줍니다."""
        if hasattr(metric_obj, "attach"):
            return metric_obj
        return IgniteMetricHandler(
            metric_fn=metric_obj,
            output_transform=from_engine(["pred", "label"]),
        )

    def _create_metrics(
        self, prefix: str = "val"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """MONAI Evaluator 규격에 맞춰 key_metric(1개)과 additional_metrics를 생성합니다."""
        base_trans = from_engine(["pred", "label"])

        def output_trans(output):
            preds, labels = base_trans(output)
            if isinstance(preds, list):
                preds = [p.float() if isinstance(p, torch.Tensor) else p for p in preds]
            elif isinstance(preds, torch.Tensor):
                preds = preds.float()
            if isinstance(labels, list):
                labels = [
                    y.float() if isinstance(y, torch.Tensor) else y for y in labels
                ]
            elif isinstance(labels, torch.Tensor):
                labels = labels.float()
            return preds, labels

        loss_metric = Loss(
            loss_fn=self.loss_function,
            output_transform=output_trans,
        )
        psnr_metric = IgniteMetricHandler(
            metric_fn=PSNRMetric(max_val=1.0),
            output_transform=output_trans,
        )
        ssim_metric = IgniteMetricHandler(
            metric_fn=SSIMMetric(spatial_dims=2, data_range=1.0),
            output_transform=output_trans,
        )

        if self.mode == "min":
            key_metric = {f"{prefix}_loss": loss_metric}
            additional_metrics = {
                f"{prefix}_psnr": psnr_metric,
                f"{prefix}_ssim": ssim_metric,
            }
        else:
            key_metric = {f"{prefix}_psnr": psnr_metric}
            additional_metrics = {
                f"{prefix}_loss": loss_metric,
                f"{prefix}_ssim": ssim_metric,
            }

        return key_metric, additional_metrics

    def _setup_tqdm_bars(self):
        epoch_pbar = tqdm(total=self.max_epochs, desc="Overall Progress", position=0)

        @self.trainer.on("STARTED")
        def on_train_start(engine):
            epoch_pbar.reset(total=self.max_epochs)

        @self.trainer.on("EPOCH_STARTED")
        def on_epoch_start(engine):
            epoch_pbar.set_description(f"Epoch {engine.state.epoch}/{self.max_epochs}")

        @self.trainer.on("EPOCH_COMPLETED")
        def on_epoch_completed(engine):
            epoch_pbar.update(1)

        @self.trainer.on("COMPLETED")
        def on_train_completed(engine):
            epoch_pbar.close()

    def _save_best_model(self, epoch: int):
        print(
            f"[+] New best model found at epoch {epoch}! Saving weights to {self.best_model_path}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
                if self.optimizer
                else None,
                "best_score": self.best_score,
            },
            self.best_model_path,
        )

        if hasattr(torch.serialization, "add_safe_globals"):
            safe_classes = {type(m) for m in self.network.modules()}
            torch.serialization.add_safe_globals(list(safe_classes))

        torch.save(self.network, self.inference_model_path)
        print(f"[+] Complete inference bundle saved to {self.inference_model_path}")

    def fit(self):
        print(f"[*] Training started. Logs & checkpoints saved to: {self.log_dir}")
        try:
            self.trainer.run()
            print("[*] Training finished!")
            self._load_and_test()
        finally:
            if hasattr(self, "writer") and self.writer is not None:
                self.writer.close()

    def train(self):
        self.fit()

    def eval(self):
        if self.evaluator is None:
            raise ValueError("Validation DataLoader is not provided.")
        print("[*] Evaluation started...")

        self.evaluator.run()

        val_metrics = self.evaluator.state.metrics
        print(f"Current Validation Metrics Result: {val_metrics}")

        print("[*] Evaluation finished!")
        return val_metrics

    def _load_and_test(self):
        if self.test_loader is None:
            print("[*] test_loader is not provided. Skipping test evaluation.")
            return

        if not os.path.exists(self.best_model_path):
            print(
                f"[-] Best model checkpoint not found at {self.best_model_path}. Skipping test."
            )
            return

        print(
            f"\n[*] Loading best model weights from {self.best_model_path} for testing..."
        )
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.network.eval()

        test_key_metric, test_additional_metrics = self._create_metrics(prefix="test")

        use_amp = self.device.type == "cuda"
        amp_kwargs = (
            {"dtype": torch.bfloat16}
            if (use_amp and torch.cuda.is_bf16_supported())
            else None
        )
        test_evaluator = SupervisedEvaluator(
            device=self.device,
            val_data_loader=self.test_loader,
            network=self.network,
            val_handlers=[],
            key_val_metric=test_key_metric,
            additional_metrics=test_additional_metrics,
            prepare_batch=self._custom_prepare_batch,
            amp=use_amp,
            amp_kwargs=amp_kwargs,
        )

        print("[*] Test evaluation started...")
        test_evaluator.run()
        test_metrics = test_evaluator.state.metrics
        print(f"[*] Test Metrics Result: {test_metrics}")

        for metric_name, metric_value in test_metrics.items():
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.item()
            self.writer.add_scalar(f"{metric_name}", metric_value, self.iteration)

        self.writer.flush()
        print("[*] Test metrics successfully logged to TensorBoard!")
