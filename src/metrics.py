import torch
import torch.nn.functional as F

def mse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    均方误差 (Mean Squared Error)
    公式: MSE = (1/n) * Σ(preds - targets)²
    """
    return F.mse_loss(preds, targets)


def rmse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    均方根误差 (Root Mean Squared Error)
    公式: RMSE = √(MSE)
    """
    return torch.sqrt(F.mse_loss(preds, targets))


def mae(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    平均绝对误差 (Mean Absolute Error)
    公式: MAE = (1/n) * Σ|preds - targets|
    """
    return F.l1_loss(preds, targets)


def mape(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    平均绝对百分比误差 (Mean Absolute Percentage Error)
    公式: MAPE = (1/n) * Σ(|preds - targets| / |targets|) * 100
    """
    epsilon = 1e-8
    return torch.mean(torch.abs((preds - targets) / (targets + epsilon))) * 100


def r2_score(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    决定系数 (Coefficient of Determination, R²)
    公式: R² = 1 - (Σ(preds - targets)² / Σ(targets - targets.mean())²)
    """
    ss_res = torch.sum((preds - targets) ** 2)
    ss_tot = torch.sum((targets - targets.mean()) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))


# 指标字典，方便统一调用
METRICS = {
    'mse': mse,
    'rmse': rmse,
    'mae': mae,
    'mape': mape,
    'r2': r2_score
}