import torch
from pathlib import Path
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    CastToTyped,
    SqueezeDimd,
)
from torch.utils.data import Subset
from monai.data import Dataset, DataLoader
from sklearn.model_selection import ShuffleSplit

from .utils import seed_worker

def get_augment(augment: list | None):
    base_transforms = [
        LoadImaged(keys=["quarter", "full"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["quarter", "full"]),
        SqueezeDimd(keys=["quarter", "full"], dim=-1),
        ScaleIntensityRanged(
            keys=["quarter", "full"],
            a_min=-100.0,
            a_max=300.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CastToTyped(keys=["quarter", "full"], dtype=torch.float32),
    ]

    if augment is not None:
        transforms = Compose(base_transforms + augment)
    else:
        transforms = Compose(base_transforms)
        
    return transforms

ROOT = Path(__file__).resolve().parent.parent / "dataset" / "CT_low_dose_reconstruction_dataset/Original Data/"
def data_load(
    target_dir: str|Path = ROOT / "Full Dose/",
    input_dir: str|Path = ROOT / "Quarter Dose/",
    augment: list | None = None,
    seed: int = 42,
):
    transforms = get_augment(augment)

    target_path = sorted(list(Path(target_dir).rglob("*.IMA")))
    input_path = sorted(list(Path(input_dir).rglob("*.IMA")))

    # 2 data for test set: L192 and L506.
    full_path = [
        str(path)
        for path in target_path
        if any("L192" in p or "L506" in p for p in path.parts)
    ]
    quarter_path = [
        str(path)
        for path in input_path
        if any("L192" in p or "L506" in p for p in path.parts)
    ]

    test_path = [
        {"quarter": quarter, "full": full}
        for full, quarter in zip(full_path, quarter_path)
    ]

    test_set = Dataset(data=test_path, transform=transforms)

    # Others: train & val
    full_path = [
        str(path)
        for path in target_path
        if not any("L192" in p or "L506" in p for p in path.parts)
    ]
    quarter_path = [
        str(path)
        for path in input_path
        if not any("L192" in p or "L506" in p for p in path.parts)
    ]
    train_path = [
        {"quarter": quarter, "full": full}
        for full, quarter in zip(full_path, quarter_path)
    ]

    learnable_set = Dataset(data=train_path, transform=transforms)

    ss = ShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=seed)
    train_index, val_index = next(iter(ss.split(X=full_path)))
    train_set = Subset(learnable_set, train_index.tolist()).dataset
    val_set = Subset(learnable_set, val_index.tolist()).dataset
    #print(train_set[0])

    return train_set, val_set, test_set


def get_loader(
    train_set: Dataset,
    val_set: Dataset,
    test_set: Dataset,
    batch_size: int,
    num_workers: int,
    seed: int = 42,
    device: str = "cpu"
):
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    train_loader, val_loader, test_loader = (
        DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        ),
        DataLoader(
            dataset=val_set, batch_size=1, shuffle=False, num_workers=num_workers
        ),
        DataLoader(
            dataset=test_set, batch_size=1, shuffle=False, num_workers=num_workers
        ),
    )

    return train_loader, val_loader, test_loader
