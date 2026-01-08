import os
import sys
import logging
import datetime
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 配置类
class TrainingConfig:
    def __init__(
        self,
        device,
        epochs: int = 50,
        save_dir: str = '../results/models',
        save_interval: int = 10,
        save_best_only: bool = True,
        early_stopping_patience: int = 10,
        tensorboard_logging: bool = False,
        log_dir: str = '../logs',
        seed: int = 42,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        patience: int = 10,
        early_stopping: bool = False,
        mixed_precision: bool = False,
        metrics_list: List[str] = ['mse', 'rmse', 'r2'],
    ) -> None:
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.save_best_only = save_best_only
        self.early_stopping_patience = early_stopping_patience
        self.tensorboard_logging = tensorboard_logging
        self.log_dir = log_dir
        self.seed = seed
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.early_stopping = early_stopping
        self.mixed_precision = mixed_precision
        self.metrics_list = metrics_list
        os.makedirs(self.save_dir, exist_ok=True)
        if self.tensorboard_logging:
            os.makedirs(self.log_dir, exist_ok=True)

    def to_dict(self) -> Dict:
        return vars(self)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'TrainingConfig':
        # Filter out keys that are not in __init__ parameters
        init_params = cls.__init__.__code__.co_varnames[1:]
        filtered_dict = {k: v for k, v in config_dict.items() if k in init_params}
        return cls(**filtered_dict)

# 回调基类
class Callback:
    def on_train_start(self, trainer: 'Trainer') -> None:
        pass

    def on_train_end(self, trainer: 'Trainer') -> None:
        pass

    def on_epoch_start(self, trainer: 'Trainer') -> None:
        pass

    def on_epoch_end(self, trainer: 'Trainer') -> None:
        pass

    def on_batch_start(self, trainer: 'Trainer') -> None:
        pass

    def on_batch_end(self, trainer: 'Trainer') -> None:
        pass

# 模型保存回调
class ModelCheckpointCallback(Callback):
    def __init__(self, save_dir: str, save_interval: int = 10, save_best_only: bool = True, monitor: str = 'val_rmse', mode: str = 'min') -> None:
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, trainer: 'Trainer') -> None:
        epoch = trainer.current_epoch
        metrics = trainer.metrics

        # 保存最佳模型
        if self.save_best_only and self.monitor in metrics:
            current_score = metrics[self.monitor]
            if trainer.best_score is None:
                trainer.best_score = current_score
                
            if not hasattr(trainer, 'best_score') or (
                (self.mode == 'min' and current_score < trainer.best_score) or
                (self.mode == 'max' and current_score > trainer.best_score)
            ):
                trainer.best_score = current_score
                checkpoint_path = os.path.join(self.save_dir, 'best_model.pth')
                trainer.save_checkpoint(checkpoint_path, epoch, is_best=True)
                metric_name = self.monitor.replace('val_', '').upper()
                logger.info(f'Saved best model to {checkpoint_path} with {metric_name}: {current_score:.4f}')

        # 按间隔保存模型
        if not self.save_best_only and (epoch + 1) % self.save_interval == 0:
            checkpoint_path = os.path.join(self.save_dir, f'model_epoch_{epoch+1}.pth')
            trainer.save_checkpoint(checkpoint_path, epoch)
            logger.info(f'Saved model at epoch {epoch+1} to {checkpoint_path}')

# 早停回调
class EarlyStoppingCallback(Callback):
    def __init__(self, patience: int = 10, monitor: str = 'val_rmse', mode: str = 'min') -> None:
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop_triggered = False

    def on_epoch_end(self, trainer: 'Trainer') -> None:
        if self.early_stop_triggered:  
            return
            
        metrics = trainer.metrics

        if self.monitor not in metrics:
            logger.warning(f'Monitor metric {self.monitor} not found in metrics.')
            return

        current_score = metrics[self.monitor]

        if self.best_score is None:
            self.best_score = current_score
        elif (self.mode == 'min' and current_score < self.best_score) or (self.mode == 'max' and current_score > self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                logger.info(f'Early stopping triggered after {trainer.current_epoch+1} epochs')
                trainer.stop_training = True
                self.early_stop_triggered = True 

# TensorBoard回调
class TensorBoardCallback(Callback):
    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir + '/run_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def on_train_start(self, trainer):
        self.writer.add_text('comment', trainer.comment)
    
    def on_epoch_end(self, trainer: 'Trainer') -> None:
        epoch = trainer.current_epoch
        metrics = trainer.metrics

        for name, value in metrics.items():
            self.writer.add_scalar(name, value, epoch)

    def on_batch_end(self, trainer: 'Trainer') -> None:
        global_step = trainer.current_epoch * len(trainer.train_loader) + trainer.current_batch
        self.writer.add_scalar('train/batch_loss', trainer.current_loss, global_step)

    def on_train_end(self, trainer: 'Trainer') -> None:
        train_id, train_pred, train_true = trainer.collect_predictions(trainer.train_loader)
        val_id, val_pred, val_true = trainer.collect_predictions(trainer.val_loader)
        # test_id, test_pred, test_true = trainer.collect_predictions(trainer.test_loader)
        
        table = f"| Dataset | ID | True Value | Predicted Value | Residual |<br>"
        table += "|---------|----|------------|-----------------|----------|<br>"
        
        for i, (id_val, t, p) in enumerate(zip(train_id, train_true.numpy(), train_pred.numpy())):
            residual = p - t
            table += f"| Train   | {id_val} | {t:.4f}     | {p:.4f}          | {residual:+.4f}  |<br>"
        
        for i, (id_val, t, p) in enumerate(zip(val_id, val_true.numpy(), val_pred.numpy())):
            residual = p - t
            table += f"| Val     | {id_val} | {t:.4f}     | {p:.4f}          | {residual:+.4f}  |<br>"
        
        # for i, (id_val, t, p) in enumerate(zip(test_id, test_true.numpy(), test_pred.numpy())):
        #     residual = p - t
        #     table += f"| Test    | {id_val} | {t:.4f}     | {p:.4f}          | {residual:+.4f}  |<br>"
        
        self.writer.add_text('residual', table)
        self.writer.close()

        
# 训练器类
class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        callbacks: Optional[List[Callback]] = None,
        comment: str = ''
    ) -> None:
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion or nn.MSELoss()
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = scheduler or optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
        self.callbacks = callbacks or []
        self.comment = comment

        # 初始化设备
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # 训练状态
        self.current_epoch = 0
        self.current_batch = 0
        self.current_loss = 0.0
        self.metrics = {}
        self.best_score = None
        self.stop_training = False

        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision and torch.cuda.is_available() else None

    def save_checkpoint(self, filepath: str, epoch: int, is_best: bool = False) -> None:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
        }

        if is_best and hasattr(self, 'best_score') and self.best_score is not None:
            checkpoint['best_score'] = self.best_score
        else:
            # 保存最新的验证指标
            val_metrics = {k: v for k, v in self.metrics.items() if k.startswith('val_')}
            if val_metrics:
                checkpoint['val_metrics'] = val_metrics

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_score = checkpoint.get('best_score', None)
        logger.info(f'Loaded checkpoint from {filepath}, starting from epoch {self.current_epoch+1}')

    def train_one_epoch(self, metrics_to_compute: List[str] = ['mse', 'r2']) -> Dict[str, float]:
        from src.metrics import METRICS
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        metrics = {}

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}')

        for i, batch in enumerate(progress_bar):
            self.current_batch = i
            # 假设最后一个元素是目标值
            spec, imgs, targets = batch['spectrum'], batch['image'], batch['sugar']
            
            # 将所有输入移到设备上
            spec = spec.to(self.device)
            imgs = imgs.to(self.device)
            targets = targets.to(self.device).squeeze()

            # 调用批次开始回调
            for callback in self.callbacks:
                callback.on_batch_start(self)

            # 清零梯度
            self.optimizer.zero_grad()

            # 前向传播
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(spec, imgs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(spec, imgs)
                loss = self.criterion(outputs, targets)

            # 反向传播和优化
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # 统计损失
            running_loss += loss.item()
            self.current_loss = loss.item()

            # 收集预测和目标值
            all_preds.extend(outputs.cpu().detach().numpy())
            all_targets.extend(targets.cpu().detach().numpy())

            # 更新进度条
            progress_bar.set_postfix(loss=loss.item())

            # 调用批次结束回调
            for callback in self.callbacks:
                callback.on_batch_end(self)

        # 计算 epoch 损失
        epoch_loss = running_loss / len(self.train_loader)
        metrics['train_loss'] = epoch_loss

        # 计算指定的指标
        preds_tensor = torch.tensor(all_preds)
        targets_tensor = torch.tensor(all_targets)

        for metric_name in metrics_to_compute:
            if metric_name in METRICS:
                metric_value = METRICS[metric_name](preds_tensor, targets_tensor).item()
                metrics[f'train_{metric_name}'] = metric_value
            else:
                logger.warning(f'指标 {metric_name} 未在METRICS字典中定义，跳过计算。')

        return metrics

    def evaluate(self, loader: DataLoader, metrics_to_compute: List[str] = ['mse', 'r2']) -> Dict[str, float]:
        from src.metrics import METRICS
        self.model.eval()
        val_loss = 0.0
        
        # 在GPU上累积预测和目标值
        all_preds = []
        all_targets = []
        
        metrics = {}

        with torch.no_grad():
            for batch in loader:
                spec, imgs, targets = batch['spectrum'], batch['image'], batch['sugar']
                spec = spec.to(self.device, non_blocking=True)
                imgs = imgs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True).squeeze()

                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(spec, imgs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(spec, imgs)
                    loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                
                all_preds.append(outputs)
                all_targets.append(targets)
        
        all_preds_tensor = torch.cat(all_preds).cpu()
        all_targets_tensor = torch.cat(all_targets).cpu()

        # 计算指标
        epoch_loss = val_loss / len(loader)
        metrics['loss'] = epoch_loss

        # 计算指定的指标
        for metric_name in metrics_to_compute:
            if metric_name in METRICS:
                metric_value = METRICS[metric_name](all_preds_tensor, all_targets_tensor).item()
                metrics[metric_name] = metric_value
            else:
                logger.warning(f'metric {metric_name} is not defined, skip calculation。')

        return metrics
    
    def collect_predictions(self, loader: DataLoader):
        self.model.eval()
        
        all_ids = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in loader:
                spec, imgs, targets = batch['spectrum'], batch['image'], batch['sugar']
                sids, cids = batch['sid'], batch['cid']
                spec = spec.to(self.device, non_blocking=True)
                imgs = imgs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True).squeeze()

                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(spec, imgs)
                else:
                    outputs = self.model(spec, imgs)
                
                all_ids.extend([f"{int(s)}_{int(c)}" for s, c in zip(sids, cids)])
                all_preds.append(outputs)
                all_targets.append(targets)
        
        all_preds_tensor = torch.cat(all_preds).cpu()
        all_targets_tensor = torch.cat(all_targets).cpu()

        return all_ids, all_preds_tensor, all_targets_tensor


    def train(self) -> None:
        # 调用训练开始回调
        for callback in self.callbacks:
            callback.on_train_start(self)

        logger.info(f'Starting training for {self.config.epochs} epochs')
        logger.info(f'Using device: {self.device}')
        logger.info(f'Train data: {len(self.train_loader.dataset)} samples')
        logger.info(f'Validation data: {len(self.val_loader.dataset)} samples')
        if self.test_loader:
            logger.info(f'Test data: {len(self.test_loader.dataset)} samples')

        # 定义要计算的指标列表
        for epoch in range(self.current_epoch, self.config.epochs):
            if self.stop_training:
                break

            # 调用epoch开始回调
            for callback in self.callbacks:
                callback.on_epoch_start(self)

            # 训练一个epoch并计算指标
            train_metrics = self.train_one_epoch(metrics_to_compute=self.config.metrics_list)
            self.metrics.update(train_metrics)

            # 验证
            val_metrics = self.evaluate(self.val_loader, metrics_to_compute=self.config.metrics_list)
            val_metrics = {f'val_{k}': v for k, v in val_metrics.items()}
            self.metrics.update(val_metrics)

            # 打印指标
            metrics_str = ', '.join([f'{k}: {v:.4f}' for k, v in self.metrics.items()])
            logger.info(f'Epoch {epoch+1}/{self.config.epochs} - {metrics_str}')

            # 更新学习率
            if self.scheduler is not None and 'train_loss' in self.metrics:
                self.scheduler.step(self.metrics['train_loss'])

            # 调用epoch结束回调
            for callback in self.callbacks:
                callback.on_epoch_end(self)

            self.current_epoch += 1

        # 最终测试
        # if self.test_loader:
        #     logger.info('Final evaluation on test set')
        #     final_test_metrics = self.evaluate(self.test_loader, metrics_to_compute=self.config.metrics_list)
        #     final_test_metrics = {f'test_{k}': v for k, v in final_test_metrics.items()}
        #     metrics_str = ', '.join([f'{k}: {v:.4f}' for k, v in final_test_metrics.items()])
        #     logger.info(f'Final test metrics - {metrics_str}')

        # 调用训练结束回调
        for callback in self.callbacks:
            callback.on_train_end(self)

        if hasattr(self, 'best_score') and self.best_score is not None:
            logger.info(f'Training completed. Best score: {self.best_score:.4f}')