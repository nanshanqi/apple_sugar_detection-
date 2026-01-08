import numpy as np
import matplotlib.pyplot as plt
import torch 

from typing import Optional, Union, List, Tuple
from scipy.linalg import lstsq
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split, KFold

class SpectralPreprocessor:

    @staticmethod
    def snv(x: np.ndarray) -> np.ndarray:
        """
        标准正态变换 (向量化实现)
        参数:
            x: 输入光谱 (1D或2D)
        返回:
            处理后的数据 (保持原始维度)
        """
        mu = np.mean(x, axis=-1, keepdims=True)
        sd = np.std(x, axis=-1, ddof=1, keepdims=True) + 1e-8
        return ((x - mu) / sd).astype(np.float32)

    @staticmethod
    def msc(x: np.ndarray, reference: Optional[np.ndarray] = None) -> np.ndarray:
        """
        乘性散射校正
        参数:
            x: (n_samples, n_features) 的2D数组
            reference: 参考光谱，若为None则使用均值光谱
        """
        if reference is None:
            reference = np.mean(x, axis=0)
        # 向量化最小二乘求解
        coeffs = lstsq(reference[:, None], x.T, lapack_driver='gelsy')[0].T
        return (x - coeffs[:, 0:1]) / coeffs[:, 1:2]

    @staticmethod
    def detrend(x: np.ndarray, type: str = "linear") -> np.ndarray:
        """
        基线去除
        参数:
            type: 'linear' 或 'constant'
        """
        from scipy.signal import detrend as sp_detrend
        if x.ndim == 1:
            return sp_detrend(x, type=type).astype(np.float32)
        return np.apply_along_axis(sp_detrend, -1, x, type=type).astype(np.float32)

    @staticmethod
    def mean_centering(x: np.ndarray) -> np.ndarray:
        """
        均值中心化
        参数:
            x: 输入光谱 (1D或2D)
        返回:
            处理后的数据 (保持原始维度)
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        return (x - mean).astype(np.float32)

    @staticmethod
    def standardization(x: np.ndarray) -> np.ndarray:
        """
        标准化 (均值为0，标准差为1)
        参数:
            x: 输入光谱 (1D或2D)
        返回:
            处理后的数据 (保持原始维度)
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, ddof=1, keepdims=True) + 1e-8
        return ((x - mean) / std).astype(np.float32)

    @staticmethod
    def savitzky_golay(x: np.ndarray, window_length: int = 11, polyorder: int = 2, deriv: int = 0) -> np.ndarray:
        """
        Savitzky-Golay滤波 (平滑和导数计算)
        参数:
            x: 输入光谱 (1D或2D)
            window_length: 窗口长度
            polyorder: 多项式阶数
            deriv: 导数阶数
        返回:
            处理后的数据 (保持原始维度)
        """
        if x.ndim == 1:
            return savgol_filter(x, window_length, polyorder, deriv=deriv).astype(np.float32)
        return np.apply_along_axis(lambda y: savgol_filter(y, window_length, polyorder, deriv=deriv), -1, x).astype(np.float32)

    @staticmethod
    def derivative(x: np.ndarray, order: int = 1, delta: float = 1.0) -> np.ndarray:
        """
        计算光谱的导数
        参数:
            x: 输入光谱 (1D或2D)
            order: 导数阶数 (1或2)
            delta: 自变量的间隔
        返回:
            处理后的数据 (保持原始维度)
        """
        if order == 1:
            deriv = np.diff(x, n=1, axis=-1)
            # 保持与输入相同的维度
            return np.concatenate([deriv, np.zeros_like(deriv[..., :1])], axis=-1).astype(np.float32) / delta
        elif order == 2:
            deriv = np.diff(x, n=2, axis=-1)
            # 保持与输入相同的维度
            return np.concatenate([np.zeros_like(deriv[..., :1]), deriv, np.zeros_like(deriv[..., :1])], axis=-1).astype(np.float32) / (delta ** 2)
        else:
            raise ValueError("仅支持一阶和二阶导数")

    @staticmethod
    def minmax_range(x: np.ndarray, 
                      range: tuple = (0, 1)) -> np.ndarray:
        """
        批量范围归一化
        参数:
            range: (min, max)
        """
        min_val = np.min(x, axis=-1, keepdims=True)
        max_val = np.max(x, axis=-1, keepdims=True)
        return (x - min_val) / (max_val - min_val + 1e-8) * \
               (range[1] - range[0]) + range[0]

    @staticmethod
    def log_transform(x: np.ndarray, 
                     offset: float = 1.0) -> np.ndarray:
        """对数变换 (天然支持向量化)"""
        return np.log10(x + offset).astype(np.float32)

    @staticmethod
    def pipeline(x: Union[np.ndarray, list], 
                steps: list = ["snv"]) -> np.ndarray:
        """
        高性能预处理流水线 
        支持的方法列表:
            'snv'           - 标准正态变换
            'msc'           - 乘性散射校正
            'detrend'       - 基线去除
            'minmax'        - 范围归一化
            'log'           - 对数变换
            'mean_center'   - 均值中心化
            'standardize'   - 标准化
            'savgol'        - Savitzky-Golay滤波
            'derivative1'   - 一阶导数
            'derivative2'   - 二阶导数
        """
        x = np.asarray(x, dtype=np.float32)
        
        # 自动处理列表输入和1D输入
        if x.ndim == 1:
            x = x[np.newaxis, :]
        
        for step in steps:
            if step == "snv":
                x = SpectralPreprocessor.snv(x)
            elif step == "msc":
                x = SpectralPreprocessor.msc(x)
            elif step == "detrend":
                x = SpectralPreprocessor.detrend(x)
            elif step == "normalize":
                x = SpectralPreprocessor.minmax_range(x)
            elif step == "log":
                x = SpectralPreprocessor.log_transform(x)
            elif step == "mean_center":
                x = SpectralPreprocessor.mean_centering(x)
            elif step == "standardize":
                x = SpectralPreprocessor.standardization(x)
            elif step == "savgol":
                x = SpectralPreprocessor.savitzky_golay(x)
            elif step == "derivative1":
                x = SpectralPreprocessor.derivative(x, order=1)
            elif step == "derivative2":
                x = SpectralPreprocessor.derivative(x, order=2)
            elif step != "raw":
                raise ValueError(f"非支持的预处理步骤: {step} (仅支持: snv/msc/detrend/normalize/log/mean_center/standardize/savgol/derivative1/derivative2)")
        
        return x.squeeze()  # 恢复1D输入的原形状


# 数据拆分工具函数

def split_dataset(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将数据集拆分为训练集、验证集和测试集
    参数:
        X: 特征数据
        y: 标签数据
        test_size: 测试集比例
        val_size: 验证集比例 (相对于训练集)
        random_state: 随机种子
    返回:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # 先拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # 再从训练集中拆分验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(1-test_size), random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


def cross_validation(X: np.ndarray, y: np.ndarray, model, k: int = 5, random_state: int = 42) -> List[dict]:
    """
    K折交叉验证
    参数:
        X: 特征数据
        y: 标签数据
        model: 模型对象 (需实现fit和predict方法)
        k: 折数
        random_state: 随机种子
    返回:
        每一折的评估结果列表
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    results = []
    
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # 训练模型
        model.fit(X_train, y_train)
        # 预测
        y_pred = model.predict(X_val)
        
        # 计算评估指标
        from src.metrics import mse, rmse, mae, r2_score
        metrics = {
            'mse': mse(y_pred, y_val).item(),
            'rmse': rmse(y_pred, y_val).item(),
            'mae': mae(y_pred, y_val).item(),
            'r2': r2_score(y_pred, y_val).item()
        }
        results.append(metrics)
    
    return results


def plot_spectra(X: np.ndarray, wavelengths: Optional[np.ndarray] = None, title: str = "光谱图") -> None:
    """
    绘制光谱图
    参数:
        X: 光谱数据 (n_samples, n_features)
        wavelengths: 波长数组
        title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    
    if wavelengths is None:
        wavelengths = np.arange(X.shape[1])
    
    for i in range(min(10, X.shape[0])):  # 最多绘制10条光谱
        plt.plot(wavelengths, X[i], label=f"样本 {i+1}")
    
    plt.xlabel("波长")
    plt.ylabel("吸光度")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_preprocessing_comparison(X: np.ndarray, X_processed: np.ndarray, wavelengths: Optional[np.ndarray] = None, title: str = "预处理前后对比") -> None:
    """
    绘制预处理前后的光谱对比图
    参数:
        X: 原始光谱数据
        X_processed: 预处理后的光谱数据
        wavelengths: 波长数组
        title: 图表标题
    """
    plt.figure(figsize=(12, 8))
    
    if wavelengths is None:
        wavelengths = np.arange(X.shape[1])
    
    # 绘制原始光谱
    plt.subplot(2, 1, 1)
    for i in range(min(3, X.shape[0])):
        plt.plot(wavelengths, X[i], label=f"原始样本 {i+1}")
    plt.title("原始光谱")
    plt.xlabel("波长")
    plt.ylabel("吸光度")
    plt.legend()
    plt.grid(True)
    
    # 绘制预处理后的光谱
    plt.subplot(2, 1, 2)
    for i in range(min(3, X_processed.shape[0])):
        plt.plot(wavelengths, X_processed[i], label=f"预处理样本 {i+1}")
    plt.title("预处理后光谱")
    plt.xlabel("波长")
    plt.ylabel("吸光度")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def feature_selection_correlation(X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    基于相关性的特征选择
    参数:
        X: 特征数据
        y: 标签数据
        threshold: 相关系数阈值
    返回:
        选中的特征索引
    """
    # 计算每个特征与目标变量的相关系数
    correlations = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
    # 选择相关系数绝对值大于阈值的特征
    selected_indices = np.where(np.abs(correlations) > threshold)[0]
    return selected_indices

def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)