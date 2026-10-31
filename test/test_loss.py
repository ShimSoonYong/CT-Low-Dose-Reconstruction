import torch
import pytest
from ..src.loss import PSNR, SSIM  # 실제 클래스가 있는 파일 이름으로 변경해주세요

@pytest.fixture
def dummy_data_2d():
    
    torch.manual_seed(42)
    y_pred = torch.rand(2, 3, 32, 32)
    y_true = torch.rand(2, 3, 32, 32)
    return y_pred, y_true


@pytest.fixture
def dummy_data_3d():
    
    torch.manual_seed(42)
    y_pred = torch.rand(2, 1, 16, 16, 16)
    y_true = torch.rand(2, 1, 16, 16, 16)
    return y_pred, y_true


class TestPSNR:
    
    def test_psnr_identical_tensors(self, dummy_data_2d):
        
        y_pred, _ = dummy_data_2d
        psnr_fn = PSNR(max_val=1.0)
        
        score = psnr_fn(y_pred, y_pred)
        
        assert torch.all(score > 70.0)
        assert score.shape == (y_pred.shape[0],)

    def test_psnr_different_tensors(self, dummy_data_2d):
        
        y_pred, y_true = dummy_data_2d
        psnr_fn = PSNR(max_val=1.0)
        
        score = psnr_fn(y_pred, y_true)
        
        assert isinstance(score, torch.Tensor)
        assert score.shape == (2,)  # Batch size = 2
        assert not torch.isnan(score).any()

    def test_psnr_non_tensor_input(self):
        
        y_pred = [[0.1, 0.2], [0.3, 0.4]]
        y_true = [[0.1, 0.2], [0.3, 0.4]]
        psnr_fn = PSNR(max_val=1.0)
        
        score = psnr_fn(y_pred, y_true)
        assert isinstance(score, torch.Tensor)


class TestSSIM:
    
    def test_ssim_identical_tensors_2d(self, dummy_data_2d):
        y_pred, _ = dummy_data_2d
        ssim_fn = SSIM(spatial_dims=2, max_val=1.0)
        score = ssim_fn(y_pred, y_pred)
        
        assert torch.allclose(score, torch.ones_like(score), atol=1e-5)
        # 배치(2)와 채널(3)을 모두 포함하는 shape 검증 (또는 mean 처리된 스칼라/배치 차원 확인)
        assert score.shape == (y_pred.shape[0], y_pred.shape[1])

    def test_ssim_identical_tensors_3d(self, dummy_data_3d):
        y_pred, _ = dummy_data_3d
        ssim_fn = SSIM(spatial_dims=3, max_val=1.0)
        score = ssim_fn(y_pred, y_pred)
        
        assert torch.allclose(score, torch.ones_like(score), atol=1e-5)
        assert score.shape == (y_pred.shape[0], y_pred.shape[1])

    def test_ssim_different_tensors_2d(self, dummy_data_2d):
        y_pred, y_true = dummy_data_2d
        ssim_fn = SSIM(spatial_dims=2, max_val=1.0)
        score = ssim_fn(y_pred, y_true)
        
        assert isinstance(score, torch.Tensor)
        assert score.shape == (y_pred.shape[0], y_pred.shape[1])
        assert torch.all(score >= -1.0) and torch.all(score <= 1.0)