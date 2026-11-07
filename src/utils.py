import os
import torch
import random
import argparse
import importlib
import numpy as np
from argparse import Namespace
from monai.transforms import Transform
from typing import List, Dict, Any, Union

# 1. 레지스트리 및 각 증강별 하드코딩된 기본 파라미터(kwargs) 정의
TRANSFORM_REGISTRY = {}


def _init_registry():
    global TRANSFORM_REGISTRY
    transforms_module = importlib.import_module("monai.transforms")

    # 각 증강 기법과 사용할 기본 파라미터(keys, 범위, 확률 등)를 매핑합니다.
    definitions = {
        "RandAffined": {
            "cls": getattr(transforms_module, "RandAffined", None),
            "kwargs": {
                "keys": ["quarter", "full"],
                "rotate_range": (-0.17, 0.17),
                "translate_range": (4, 4),
                "scale_range": (0.1, 0.1),
                "prob": 0.5,
                "padding_mode": "reflection",
                "mode": "bilinear",
            },
        },
        "RandRot90d": {
            "cls": getattr(transforms_module, "RandRotate90d", None),
            "kwargs": {"keys": ["quarter", "full"], "prob": 0.5, "max_k": 3},
        },
        "RandFlipd_0": {
            "cls": getattr(transforms_module, "RandFlipd", None),
            "kwargs": {"keys": ["quarter", "full"], "prob": 0.5, "spatial_axis": 0},
        },
        "RandFlipd_1": {
            "cls": getattr(transforms_module, "RandFlipd", None),
            "kwargs": {"keys": ["quarter", "full"], "prob": 0.5, "spatial_axis": 1},
        },
        "RandScaleIntensityd": {
            "cls": getattr(transforms_module, "RandScaleIntensityd", None),
            "kwargs": {"keys": ["quarter", "full"], "factors": 0.1, "prob": 0.5},
        },
        "RandShiftIntensityd": {
            "cls": getattr(transforms_module, "RandShiftIntensityd", None),
            "kwargs": {"keys": ["quarter", "full"], "offsets": 0.1, "prob": 0.5},
        },
        "RandGaussianNoised": {
            "cls": getattr(transforms_module, "RandGaussianNoised", None),
            "kwargs": {
                "keys": ["quarter"],
                "mean": 0.0,
                "std": 0.02,
                "prob": 0.3,
            },
        },
        "RandGaussianSmoothd": {
            "cls": getattr(transforms_module, "RandGaussianSmoothd", None),
            "kwargs": {
                "keys": ["quarter", "full"],
                "sigma_x": (0.5, 1.5),
                "sigma_y": (0.5, 1.5),
                "prob": 0.3,
            },
        },
        # 보간 블러 및 HU 왜곡이 전혀 없는 초고속 무손실 직교 대칭(D4 Dihedral) 복합 증강
        "Rot90_Flip_Combo": {
            "composite": [
                (
                    "RandRotate90d",
                    {"keys": ["quarter", "full"], "prob": 0.5, "max_k": 3},
                ),
                (
                    "RandFlipd",
                    {"keys": ["quarter", "full"], "prob": 0.5, "spatial_axis": 0},
                ),
                (
                    "RandFlipd",
                    {"keys": ["quarter", "full"], "prob": 0.5, "spatial_axis": 1},
                ),
            ]
        },
        # 복합 증강들 등록 (하위 호환 유지; 노이즈는 입력 quarter에만 인가)
        "Affine_Noise_Combo": {
            "composite": [
                (
                    "RandAffined",
                    {
                        "keys": ["quarter", "full"],
                        "rotate_range": (-0.17, 0.17),
                        "prob": 0.5,
                        "mode": "bilinear",
                    },
                ),
                (
                    "RandGaussianNoised",
                    {
                        "keys": ["quarter"],
                        "mean": 0.0,
                        "std": 0.02,
                        "prob": 0.3,
                    },
                ),
            ]
        },
        "Rot90_Scale_Combo": {
            "composite": [
                (
                    "RandRotate90d",
                    {"keys": ["quarter", "full"], "prob": 0.5, "max_k": 3},
                ),
                (
                    "RandScaleIntensityd",
                    {"keys": ["quarter", "full"], "factors": 0.1, "prob": 0.5},
                ),
            ]
        },
    }

    for name, info in definitions.items():
        if "composite" in info:
            # 복합 증강인 경우 내부 항목들을 미리 세팅
            composite_list = []
            for c_name, c_kwargs in info["composite"]:
                c_cls = getattr(transforms_module, c_name, None)
                if c_cls:
                    composite_list.append(c_cls(**c_kwargs))
            TRANSFORM_REGISTRY[name] = {"composite": composite_list}
        else:
            if info["cls"] is not None:
                TRANSFORM_REGISTRY[name] = info


# 모듈 로드 시 레지스트리 초기화
_init_registry()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_augs(augments: List[str]) -> List[Transform]:
    """
    레지스트리 키를 기반으로 Transform 객체 리스트를 안전하게 생성합니다.
    """
    aug_list: List[Transform] = []

    for aug_name in augments:
        if not aug_name or str(aug_name).lower() == "none":
            continue

        if aug_name not in TRANSFORM_REGISTRY:
            raise AttributeError(
                f"monai.transforms에 지원하지 않거나 등록되지 않은 Transform 이름입니다: {aug_name}"
            )

        reg_item = TRANSFORM_REGISTRY[aug_name]

        # 복합 증강(Combo)인 경우
        if "composite" in reg_item:
            aug_list.extend(reg_item["composite"])
        # 단일 증강인 경우
        else:
            transform_cls = reg_item["cls"]
            kwargs = reg_item["kwargs"]
            aug_list.append(transform_cls(**kwargs))

    return aug_list


def get_args() -> Namespace:
    """
    터미널 인자를 파싱하여 학습 및 추론에 필요한 설정값을 반환합니다.
    """
    parser = argparse.ArgumentParser()

    # --- 1. 기본 학습 설정 ---
    parser.add_argument("--model", type=str, help="테스트할 모델")
    parser.add_argument(
        "--epochs", type=int, default=20, help="최대 학습 에포크 수 (기본값: 20)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="미니배치 크기 (기본값: 4)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="학습률(Learning Rate) (기본값: 1e-4)"
    )
    parser.add_argument(
        "--grad_norm",
        type=float,
        default=1.0,
        help="그레디언트 노름의 상한 (기본값: 1.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="사용할 디바이스 지정 (기본값: cuda)",
    )
    parser.add_argument(
        "--augment",
        nargs="+",
        default=["none"],
        help="사용할 데이터 증강 기법 (레지스트리 키 이름)",
    )

    # --- 2. 경로 및 체크포인트 설정 ---
    parser.add_argument(
        "--data_root",
        type=str,
        default="dataset/CT_low_dose_reconstruction_dataset/Original Data/",
        help="데이터세트들의 루트 디렉토리",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="Full Dose",
        help="full dose 데이터세트의 디렉토리",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="Quarter Dose",
        help="quarter dose 데이터세트의 디렉토리",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="./log",
        help="로그 및 모델 체크포인트가 저장될 디렉터리 경로",
    )

    # --- 3. Early Stopping 및 검증 설정 ---
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="조기 종료(Early Stopping) 허용 에포크 수 (기본값: 10)",
    )
    parser.add_argument(
        "--min_delta", type=float, default=1e-4, help="성능 개선으로 인정할 최소 변화량"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="min",
        choices=["min", "max"],
        help="성능 지표 최적화 방향 (손실 기준이면 min, PSNR/SSIM이면 max)",
    )

    # --- 4. 기타 유틸리티 설정 ---
    parser.add_argument(
        "--seed", type=int, default=42, help="재현성을 위한 랜덤 시드 번호 (기본값: 42)"
    )

    return parser.parse_args()
