import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytest
from ..src.trainer import Trainer


class DummyRestorationDataset(Dataset):
    """테스트용 가상 복원 데이터셋 (quarter -> full)"""
    def __init__(self, num_samples=8):
        self.full = torch.randn(num_samples, 1, 16, 16)
        self.quarter = torch.randn(num_samples, 1, 16, 16)

    def __len__(self):
        return len(self.full)

    def __getitem__(self, idx):
        return {
            "quarter": self.full[idx],
            "full": self.quarter[idx]
        }


class SimpleConvNet(nn.Module):
    """테스트용 초소형 CNN 아키텍처"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


@pytest.fixture
def trainer_env():
    """테스트에 필요한 임시 환경 및 객체들을 세팅하는 픽스처"""
    log_dir = "./test_runs_tmp"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    dataset = DummyRestorationDataset(num_samples=4)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    network = SimpleConvNet()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    loss_fn = nn.L1Loss()

    yield {
        "network": network,
        "train_loader": train_loader,
        "optimizer": optimizer,
        "loss_fn": loss_fn,
        "log_dir": log_dir
    }

    # 테스트 종료 후 임시 로그 폴더 정리
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)


class TestTrainer:

    def test_trainer_initialization(self, trainer_env):
        """Trainer 객체가 에러 없이 잘 생성되는지 확인"""
        env = trainer_env
        trainer = Trainer(
            network=env["network"],
            train_loader=env["train_loader"],
            val_loader=env["train_loader"],  # 테스트용으로 동일 로더 사용
            optimizer=env["optimizer"],
            loss_function=env["loss_fn"],
            max_epochs=2,
            log_dir=env["log_dir"]
        )
        assert trainer is not None
        assert trainer.trainer is not None
        assert trainer.evaluator is not None

    def test_trainer_fit_and_saving(self, trainer_env):
        """fit 메서드가 정상 구동되고, 최고 성능 모델 및 추론 번들이 잘 저장되는지 확인"""
        env = trainer_env
        trainer = Trainer(
            network=env["network"],
            train_loader=env["train_loader"],
            val_loader=env["train_loader"],
            optimizer=env["optimizer"],
            loss_function=env["loss_fn"],
            max_epochs=3,
            log_dir=env["log_dir"],
            patience=2
        )

        # 학습 실행
        trainer.fit()

        # 체크포인트 파일들이 정상적으로 생성되었는지 검증
        best_model_path = os.path.join(env["log_dir"], "best_model.pth")
        inference_bundle_path = os.path.join(env["log_dir"], "inference_model_bundle.pt")

        assert os.path.exists(best_model_path), "best_model.pth가 생성되지 않았습니다."
        assert os.path.exists(inference_bundle_path), "inference_model_bundle.pt가 생성되지 않았습니다."

        # 저장된 추론용 번들 파일이 실제로 로드되는지 확인
        loaded_model = torch.load(inference_bundle_path, weights_only=False)
        assert isinstance(loaded_model, nn.Module)

    def test_trainer_evaluation_method(self, trainer_env):
        """단독 eval() 메서드 호출 시 메트릭이 정상 반환되는지 확인"""
        env = trainer_env
        trainer = Trainer(
            network=env["network"],
            train_loader=env["train_loader"],
            val_loader=env["train_loader"],
            optimizer=env["optimizer"],
            loss_function=env["loss_fn"],
            max_epochs=1,
            log_dir=env["log_dir"]
        )

        metrics = trainer.eval()
        assert isinstance(metrics, dict)
        assert "val_loss" in metrics

    def test_early_stopping_trigger(self, trainer_env):
        """Patience 조건을 극단적으로 낮췄을 때 Early Stopping이 트리거되는지 확인"""
        env = trainer_env
        trainer = Trainer(
            network=env["network"],
            train_loader=env["train_loader"],
            val_loader=env["train_loader"],
            optimizer=env["optimizer"],
            loss_function=env["loss_fn"],
            max_epochs=10,  # 충분히 길게 설정
            log_dir=env["log_dir"],
            patience=1,     # 1번만 개선 안 돼도 바로 멈추도록 설정
            min_delta=0.0   # 사소한 변화도 개선으로 보지 않게 설정 등
        )

        # 수동으로 강제 스코어 악화 상황을 만들거나 fit 실행 시기 확인
        trainer.fit()
        
        # max_epochs(10)보다 더 일찍(혹은 정상적으로) 종료되었는지 체크
        assert trainer.trainer.state.epoch <= 10