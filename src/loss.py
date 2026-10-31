import torch
import torch.nn.functional as F
from ignite.metrics import Metric
from monai.losses import SSIMLoss
from typing import Union, Sequence
from ignite.exceptions import NotComputableError


def _unpack_metric_output(output):
    """MONAI SupervisedEvaluator의 출력(list of dicts, dict, tuple)에서 pred와 label 텐서를 추출합니다."""
    if isinstance(output, (list, tuple)):
        if len(output) > 0 and isinstance(output[0], dict):
            y_pred = torch.stack([item.get("pred", item.get("image")) for item in output])
            y = torch.stack([item.get("label", item.get("high_res", item.get("full"))) for item in output])
            return y_pred, y
        elif len(output) == 2:
            return output[0], output[1]
    elif isinstance(output, dict):
        if "pred" in output and "label" in output:
            return output["pred"], output["label"]
        values = list(output.values())
        return values[0], values[1]
    raise TypeError(f"Unsupported output format for metric update: {type(output)}")


class PSNR(Metric):
    def __init__(
        self,
        max_val: float = 1.0,
        output_transform=lambda x: x,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__(output_transform=output_transform, device=device)
        self.max_val = max_val
        self.reset()

    def reset(self) -> None:
        self._sum = 0.0
        self._num_examples = 0

    def _compute_psnr(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tensor(y_pred, device=y.device if isinstance(y, torch.Tensor) else None)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=y_pred.device)

        mse = torch.mean((y_pred - y) ** 2, dim=list(range(1, y_pred.ndim)))
        max_v = torch.tensor(self.max_val, dtype=y_pred.dtype, device=y_pred.device)
        psnr = 20 * torch.log10(max_v) - 10 * torch.log10(mse + 1e-8)
        return psnr

    def update(self, output) -> None:
        y_pred, y = _unpack_metric_output(output)
        psnr = self._compute_psnr(y_pred, y)
        self._sum += psnr.sum().item()
        self._num_examples += y_pred.shape[0]

    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("PSNR must have examples before it can be computed.")
        return self._sum / self._num_examples

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._compute_psnr(y_pred, y)


class SSIM(Metric):
    def __init__(
        self,
        spatial_dims: int = 2,
        max_val: float = 1.0,
        kernel_size: Union[int, Sequence[int]] = 11,
        output_transform=lambda x: x,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__(output_transform=output_transform, device=device)
        self.spatial_dims = spatial_dims
        self.max_val = max_val
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, (list, tuple))
            else (kernel_size,) * spatial_dims
        )
        self.reset()

    def reset(self) -> None:
        self._sum = 0.0
        self._num_examples = 0

    def _gaussian_kernel(
        self, channels: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        sigma = 1.5
        kernels = []
        for k_size in self.kernel_size:
            coords = torch.arange(k_size, dtype=dtype, device=device)
            coords -= (k_size - 1) / 2.0
            g = torch.exp(-(coords**2) / (2 * sigma**2))
            g /= g.sum()
            kernels.append(g)

        if self.spatial_dims == 2:
            k_h, k_w = self.kernel_size
            kernel = torch.ger(kernels[0], kernels[1])
            kernel = kernel.expand(channels, 1, k_h, k_w)
        elif self.spatial_dims == 3:
            k_d, k_h, k_w = self.kernel_size
            kernel = (
                kernels[0].view(-1, 1, 1)
                * kernels[1].view(1, -1, 1)
                * kernels[2].view(1, 1, -1)
            )
            kernel = kernel.expand(channels, 1, k_d, k_h, k_w)
        else:
            raise ValueError("Only 2D and 3D spatial dimensions are supported.")

        return kernel

    def _compute_ssim(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tensor(y_pred, device=y.device if isinstance(y, torch.Tensor) else None)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=y_pred.device)

        b, c = y_pred.shape[:2]
        device = y_pred.device
        dtype = y_pred.dtype

        C1 = (0.01 * self.max_val) ** 2
        C2 = (0.03 * self.max_val) ** 2

        kernel = self._gaussian_kernel(c, device, dtype)

        if self.spatial_dims == 2:
            conv = lambda x: F.conv2d(
                x, kernel, padding=self.kernel_size[0] // 2, groups=c
            )
        elif self.spatial_dims == 3:
            conv = lambda x: F.conv3d(
                x, kernel, padding=self.kernel_size[0] // 2, groups=c
            )
        else:
            raise ValueError("Only 2D and 3D spatial dimensions are supported.")

        mu1 = conv(y_pred)
        mu2 = conv(y)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = conv(y_pred**2) - mu1_sq
        sigma2_sq = conv(y**2) - mu2_sq
        sigma12 = conv(y_pred * y) - mu1_mu2

        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = numerator / (denominator + 1e-8)
        return ssim_map.mean(dim=list(range(2, ssim_map.ndim)))

    def update(self, output) -> None:
        y_pred, y = _unpack_metric_output(output)
        ssim_score = self._compute_ssim(y_pred, y)
        self._sum += ssim_score.sum().item()
        self._num_examples += y_pred.shape[0]

    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("SSIM must have examples before it can be computed.")
        return self._sum / self._num_examples

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._compute_ssim(y_pred, y)