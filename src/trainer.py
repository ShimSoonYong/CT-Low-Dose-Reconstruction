import os
import torch
import torch.nn as nn
from pathlib import Path
from ignite.metrics import Loss
from torch.utils.data import DataLoader
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
        loss_function: Optional[nn.Module] = None,
        max_epochs: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        log_dir: str | Path = "./log/restoration",
        key_val_metric: Optional[Dict[str, Any]] = None,
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "min",
    ):
        self.network = network
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_function = loss_function or nn.L1Loss()
        self.max_epochs = max_epochs
        self.device = torch.device(device)
        self.log_dir = str(log_dir)

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.patience_counter = 0
        self.best_model_path = os.path.join(self.log_dir, "best_model.pth")
        self.inference_model_path = os.path.join(self.log_dir, "inference_model_bundle.pt")

        os.makedirs(self.log_dir, exist_ok=True)
        self.network.to(self.device)

        # 1. MONAI/Ignite 공식 문서 스펙에 따른 배치 데이터 파싱 함수 (prepare_batch)
        def custom_prepare_batch(batchdata, device, non_blocking):
            if isinstance(batchdata, list):
                batchdata = batchdata[0]
            
            # 데이터셋 키 규격("quarter"/"full" 또는 "low_res"/"high_res") 안전하게 대응
            low_res = batchdata.get("quarter", batchdata.get("low_res"))
            high_res = batchdata.get("full", batchdata.get("high_res"))
            
            if low_res is None or high_res is None:
                raise KeyError(f"Batch keys not found. Available keys: {list(batchdata.keys())}")
                
            return (
                low_res.to(device=device, non_blocking=non_blocking),
                high_res.to(device=device, non_blocking=non_blocking)
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

        self.eval_handlers = [
            TensorBoardStatsHandler(
                log_dir=self.log_dir,
            ),
        ]

        # 3. SupervisedTrainer 설정 (prepare_batch 반영)
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

        # 4. SupervisedEvaluator 설정 (prepare_batch 반영 및 메트릭 보장)
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
                print(f"\n[Epoch {current_epoch}] Running validation...")
                
                self.evaluator.state.epoch = current_epoch
                self.evaluator.state.iteration = engine.state.iteration
                
                self.evaluator.run()

                val_metrics = self.evaluator.state.metrics
                print(f"[Epoch {current_epoch}] Validation Metrics Result: {val_metrics}")
                
                current_score = val_metrics.get("val_loss", 0.0)
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

    def _save_best_model(self, epoch: int):
        print(f"[+] New best model found at epoch {epoch}! Saving weights to {self.best_model_path}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
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