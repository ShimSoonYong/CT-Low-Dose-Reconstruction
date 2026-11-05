import os
import torch
import random
import numpy as np
import pytest
from argparse import Namespace
from monai.transforms import Transform

from ..src.utils import seed_everything, seed_worker, get_augs, get_args


class TestSeedUtilities:
    """랜덤 시드 고정 및 워커 관련 유틸리티 테스트"""

    def test_seed_everything_reproducibility(self):
        """seed_everything 호출 시 Python, NumPy, PyTorch 시드가 동일하게 고정되는지 확인"""
        seed_everything(42)
        py_val_1 = random.random()
        np_val_1 = np.random.rand()
        torch_val_1 = torch.rand(1).item()

        seed_everything(42)
        py_val_2 = random.random()
        np_val_2 = np.random.rand()
        torch_val_2 = torch.rand(1).item()

        assert py_val_1 == py_val_2
        assert np_val_1 == np_val_2
        assert torch_val_1 == torch_val_2

    def test_seed_everything_env_var(self):
        """PYTHONHASHSEED 환경 변수가 올바르게 설정되는지 확인"""
        seed_everything(123)
        assert os.environ["PYTHONHASHSEED"] == "123"


class TestGetAugs:
    """getattr 기반 동적 증강 생성 함수 테스트"""

    def test_get_augs_with_string_inputs(self):
        """문자열 리스트로 전달된 증강 이름이 올바른 MONAI Transform 객체로 변환되는지 확인"""
        augment_names = ["RandFlipd_0", "RandGaussianNoised"]
        transforms = get_augs(augment_names)

        assert isinstance(transforms, list)
        assert len(transforms) == 2
        for t in transforms:
            assert isinstance(t, Transform)

    def test_get_augs_with_none_input(self):
        """'none'이 포함된 경우 무시하고 빈 리스트 혹은 유효한 것만 반환하는지 확인"""
        transforms = get_augs(["none"])
        assert len(transforms) == 0

    def test_get_augs_invalid_name(self):
        """존재하지 않는 Transform 이름을 넣었을 때 AttributeError가 발생하는지 확인"""
        with pytest.raises(AttributeError, match="monai.transforms에"):
            get_augs(["NonExistentTransformd"])


class TestGetArgs:
    """argparse 기반 터미널 인자 파싱 테스트"""

    def test_get_args_defaults(self, monkeypatch):
        """인자를 주지 않고 실행했을 때 기본값(Default)들이 올바르게 세팅되는지 확인"""
        monkeypatch.setattr("sys.argv", ["script.py"])

        args = get_args()

        assert isinstance(args, Namespace)
        assert args.epochs == 20
        assert args.batch_size == 4
        assert args.lr == 1e-4
        assert args.device == "cuda"
        assert args.augment == ["none"]
        assert args.log_dir == "./log"
        assert args.patience == 10
        assert args.min_delta == 1e-4
        assert args.mode == "min"
        assert args.seed == 42

    def test_get_args_custom_inputs(self, monkeypatch):
        """터미널에서 커스텀 인자들을 넘겼을 때 파싱이 정상적으로 이루어지는지 확인"""
        monkeypatch.setattr(
            "sys.argv",
            [
                "script.py",
                "--epochs",
                "20",
                "--batch_size",
                "8",
                "--augment",
                "RandFlipd",
                "RandGaussianNoised",
                "--mode",
                "max",
            ],
        )

        args = get_args()

        assert args.epochs == 20
        assert args.batch_size == 8
        assert args.augment == ["RandFlipd", "RandGaussianNoised"]
        assert args.mode == "max"
