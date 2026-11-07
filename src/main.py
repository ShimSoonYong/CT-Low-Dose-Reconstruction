from numpy import test
import torch
from pathlib import Path
from torch import nn, optim

from .trainer import Trainer
from .loss import PSNR, SSIM, SSIMLoss
from .data import data_load, get_loader
from .utils import get_args, seed_everything, get_augs
from .UNet import U_Net, R2U_Net, R2AttU_Net, AttU_Net, NestedUNet, U_Net_NoRes


def main(args):
    # 1. 시드 고정
    seed_everything(getattr(args, "seed", 42))

    # 2. 증강(Augmentations) 객체 호출 (필요시 args 활용)
    augs = get_augs(args.augment)

    # 3. 디바이스 설정
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not getattr(args, "no_cuda", False)
        else "cpu"
    )
    print(f"[*] Using device: {device}")

    # 4. 데이터셋 디렉토리 경로 설정
    root = Path(__file__).resolve().parent.parent / getattr(
        args, "data_root", "dataset/CT_low_dose_reconstruction_dataset/Original Data/"
    )

    target_dir = root / getattr(args, "target_dir", "Full Dose/")
    input_dir = root / getattr(args, "input_dir", "Quarter Dose/")

    print(
        f"[*] Loading datasets from:\n - Full Dose (Target): {target_dir}\n - Quarter Dose (Input): {input_dir}"
    )

    # 5. 데이터셋 로드 및 데이터로더 생성
    train_set, val_set, test_set = data_load(
        target_dir, input_dir, augment=augs, seed=args.seed
    )

    batch_size = getattr(args, "batch_size", 4)
    num_workers = getattr(args, "num_workers", 12)

    train_loader, val_loader, test_loader = get_loader(
        train_set,
        val_set,
        test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=args.seed,
    )

    # 6. 모델(네트워크) 선택 및 초기화
    model_name = getattr(args, "model", "U_Net")
    in_channels = getattr(args, "in_channels", 1)
    out_channels = getattr(args, "out_channels", 1)

    if model_name == "U_Net":
        network = U_Net(in_ch=in_channels, out_ch=out_channels)
    elif model_name == "U_Net_NoRes":
        network = U_Net_NoRes(in_ch=in_channels, out_ch=out_channels)
    elif model_name == "R2U_Net":
        network = R2U_Net(img_ch=in_channels, output_ch=out_channels)
    elif model_name == "AttU_Net":
        network = AttU_Net(img_ch=in_channels, output_ch=out_channels)
    elif model_name == "R2AttU_Net":
        network = R2AttU_Net(in_ch=in_channels, out_ch=out_channels)
    elif model_name == "NestedUNet":
        network = NestedUNet(in_ch=in_channels, out_ch=out_channels)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    network.to(device)
    network.compile()

    # 7. 옵티마이저 및 손실 함수(Loss) 설정
    max_epochs = getattr(args, "epochs", 10)
    lr = getattr(args, "lr", 1e-4)
    optimizer = optim.AdamW(network.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=max(1, max_epochs // 10), eta_min=1e-6
    )
    l1_criterion = nn.L1Loss()
    ssim_criterion = SSIMLoss(spatial_dims=2, data_range=1.0)

    def loss_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # AMP autocast 내부에서도 F.conv2d가 FP16으로 다운캐스팅되어 SSIM 분산 계산 시
        # 0/0 NaN이 발생하는 것을 방지하기 위해 손실 계산은 명시적으로 FP32에서 수행
        with torch.amp.autocast(device_type=device.type, enabled=False):
            x_fp32 = x.float()
            y_fp32 = y.float()
            return l1_criterion(x_fp32, y_fp32) + 0.1 * ssim_criterion(
                x_fp32.clamp(-1.0, 2.0), y_fp32
            )

    # 8. 검증 평가지표(Metric)는 Trainer 내부에서 자동 생성

    # 9. Trainer 초기화 및 학습(fit) 실행
    log_dir = Path(getattr(args, "log_dir", "./log")) / Path(
        model_name + "_" + "".join(args.augment)
    )
    patience = getattr(args, "patience", 5)

    trainer = Trainer(
        network=network,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        loss_function=loss_function,
        max_epochs=max_epochs,
        device=str(device),
        log_dir=log_dir,
        patience=patience,
        max_grad_norm=args.grad_norm,
        mode="max",
    )

    print("[*] Fitting")
    trainer.fit()
    print("[*] Fitting complete")


if __name__ == "__main__":
    seed_everything()
    args = get_args()
    main(args)
    print("Finished")
