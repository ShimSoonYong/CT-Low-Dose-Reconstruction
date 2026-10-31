import os
from pathlib import Path
from monai.data import MetaTensor
from ..src.data import get_loader, data_load


def test_path():
    root = Path(__file__).resolve().parent.parent

    target_dir = root / "dataset/CT_low_dose_reconstruction_dataset/Original Data/Full Dose/"
    input_dir = root / "dataset/CT_low_dose_reconstruction_dataset/Original Data/Quarter Dose/"

    target_path = list(Path(target_dir).rglob("*.IMA"))
    input_path = list(Path(input_dir).rglob("*.IMA"))
    assert Path(os.getcwd()) == root, f"Directory is {os.getcwd()}"
    assert next(iter(target_path)) is not None, "No IMA data in target path"
    assert next(iter(input_path)) is not None, "No IMA data in input path"


def test_load():

    train_set, val_set, test_set = data_load()
    
    train_loader, val_loader, test_loader = get_loader(
        train_set, val_set, test_set, batch_size=100, num_workers=16
    )

    data = next(iter(train_loader))
    input = data["quarter"]
    target = data["full"]
    print(input.shape)

    assert isinstance(input, MetaTensor), (
        "input data did not transformed to lazy Tensor properly"
    )
    assert isinstance(target, MetaTensor), (
        "target data did not transformed to lazy Tensor properly"
    )
