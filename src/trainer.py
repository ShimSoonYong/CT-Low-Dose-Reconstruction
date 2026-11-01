import os
import torch
import torch.nn as nn
from pathlib import Path
from ignite.metrics import Loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.handlers import (
    TensorBoardStatsHandler,
    TensorBoardImageHandler,
    CheckpointSaver,
    from_engine,
)
from typing import Optional, Dict, Any
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        network: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        loss_function: Optional[nn.Module] = None,
        max_epochs: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        log_dir: str | Path = "./log/run",
        key_val_metric: Optional[Dict[str, Any]] = None,
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "min",
        max_grad_norm: float = 1.0,
    ):
        self.network = network
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler  # <-- 스케줄러 저장
        self.loss_function = loss_function or nn.L1Loss()
        self.max_epochs = max_epochs
        self.device = torch.device(device)
        self.log_dir = str(log_dir)
        self.max_grad_norm = max_grad_norm

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

        # TensorBoard 수동 로깅을 위한 SummaryWriter 초기화
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # 1. 배치 데이터 파싱 함수 (prepare_batch)
        def custom_prepare_batch(batchdata, device, non_blocking):
            if isinstance(batchdata, list):
                batchdata = batchdata[0]

            low_res = batchdata.get("quarter", batchdata.get("low_res"))
            high_res = batchdata.get("full", batchdata.get("full"))

            if low_res is None or high_res is None:
                raise KeyError(
                    f"Batch keys not found. Available keys: {list(batchdata.keys())}"
                )

            return (
                low_res.to(device=device, non_blocking=non_blocking),
                high_res.to(device=device, non_blocking=non_blocking),
            )

        # 2. 텐서보드 이미지 로깅을 위한 배치 변환 함수 (batch_transform)
        def custom_batch_transform(batch):
            if isinstance(batch, list):
                batch = batch[0]
            low_res = batch.get("quarter", batch.get("low_res"))
            high_res = batch.get("full", batch.get("high_res"))
            return low_res, high_res

        self.train_handlers = [
            TensorBoardStatsHandler(
                tag_name="train_loss",
                output_transform=from_engine(["loss"], first=True),
                log_dir=self.log_dir,
            ),
            TensorBoardImageHandler(
                batch_transform=custom_batch_transform,
                output_transform=from_engine(["pred"]),
                max_channels=1,
                max_frames=4,
                epoch_level=True,
                log_dir=self.log_dir,
            ),
            CheckpointSaver(
                save_dir=self.log_dir,
                save_dict={"network": self.network},
                save_interval=1,
                save_final=True,
            ),
        ]

        self.eval_handlers = []

        # 3. SupervisedTrainer 설정
        self.trainer = SupervisedTrainer(
            device=self.device,
            max_epochs=self.max_epochs,
            network=self.network,
            train_data_loader=self.train_loader,
            optimizer=self.optimizer,
            loss_function=self.loss_function,
            train_handlers=self.train_handlers,
            prepare_batch=custom_prepare_batch,
            amp=True,
        )

        @self.trainer.on("BACKWARD_COMPLETED")
        def grad_clip_handler(engine):
            clip_grad_norm_(self.network.parameters(), max_norm=self.max_grad_norm)

        # 4. SupervisedEvaluator 설정
        self.evaluator = None
        if self.val_loader is not None:
            if key_val_metric is None:
                val_loss_metric = Loss(
                    loss_fn=self.loss_function,
                    output_transform=from_engine(["pred", "label"]),
                )
                key_val_metric = {"val_loss": val_loss_metric}

            self.evaluator = SupervisedEvaluator(
                device=self.device,
                val_data_loader=self.val_loader,
                network=self.network,
                val_handlers=self.eval_handlers,
                key_val_metric=key_val_metric,
                prepare_batch=custom_prepare_batch,
                amp=True,
            )

            @self.trainer.on("EPOCH_COMPLETED")
            def run_evaluation_and_checkpoint(engine):
                current_epoch = engine.state.epoch
                current_iteration = engine.state.iteration
                print(f"\n[Epoch {current_epoch}] Running validation...")

                # [스케줄러 업데이트 처리]
                if self.lr_scheduler is not None:
                    # ReduceLROnPlateau처럼 검증 스코어(loss 등)를 인자로 받는 스케줄러 대응
                    if isinstance(
                        self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        # 임시로 검증 손실을 가져오거나 val_loss 메트릭 활용
                        val_loss_val = self.evaluator.state.metrics.get("val_loss", 0.0)
                        if isinstance(val_loss_val, torch.Tensor):
                            val_loss_val = val_loss_val.item()
                        self.lr_scheduler.step(val_loss_val)
                    else:
                        self.lr_scheduler.step()

                # 현재 학습률(Learning Rate) TensorBoard 로깅
                if self.optimizer is not None:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar(
                        "learning_rate", current_lr, current_iteration
                    )

                self.evaluator.state.epoch = current_epoch
                self.evaluator.state.iteration = current_iteration

                self.evaluator.run()

                val_metrics = self.evaluator.state.metrics
                print(
                    f"[Epoch {current_epoch}] Validation Metrics Result: {val_metrics}"
                )

                for metric_name, metric_value in val_metrics.items():
                    if isinstance(metric_value, torch.Tensor):
                        metric_value = metric_value.item()
                    self.writer.add_scalar(metric_name, metric_value, current_iteration)
                self.writer.flush()

                current_score = val_metrics.get(
                    "val_psnr", val_metrics.get("val_loss", 0.0)
                )
                if isinstance(current_score, torch.Tensor):
                    current_score = current_score.item()

                is_better = (
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
            if hasattr(self, "writer"):
                self.writer.close()

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
        self.trainer.run()
        print("[*] Training finished!")

    def train(self):
        self.fit()

    def eval(self):
        if self.evaluator is None:
            raise ValueError("Validation DataLoader is not provided.")
        print("[*] Evaluation started...")
        self.evaluator.run()
        print("[*] Evaluation finished!")
        return self.evaluator.state.metrics
