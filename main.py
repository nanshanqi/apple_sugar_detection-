import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import shutil
from torch import nn, optim
from src.model import get_model
from src.dataset import get_data_loaders
from src.trainer import TrainingConfig, Trainer, ModelCheckpointCallback, EarlyStoppingCallback, TensorBoardCallback


def clear_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或符号链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除子目录及内容
        except Exception as e:
            print(f"删除 {file_path} 时出错: {e}")

clear_folder("models")

# 模型相关
hidden_dim = 512
views = 2
model_config = {
    'spectral_encoder' : {
        'type' : 'transformer',
        # 'layers' : (2, 2),
        'output_dim' : hidden_dim,
        # 'pool_type' : 'avg'
    },
    # 'image_encoder' : {
    #     'type' : 'multiview',
    #     'base_encoder' : {
    #         'type' : 'vit',
    #         'output_dim' : hidden_dim,
    #         'pretrained' : False
    #     }
    #     'num_views' : views,
    #     'output_dim' : hidden_dim,
    #     'fusion_method' : 'attention',
    #     'dropout' : 1.0
    # },
    'image_encoder' : {
        'type' : 'resnet',
        'output_dim' : hidden_dim,
        'model_name' : 'resnet18'
        # 'pretrained' : True
    }
}

# 数据集相关
img_dir = r'data/fig'
excel_path = r'data/specv-.xlsx'
batch_size = 32
img_size = 224
seed = 0
spec_preprocess = ['snv']

model = get_model(
    encoder_config = model_config,
    output_dim = 1,
    dropout = 0.1,
    output_activation=None,
    fusion_method='concat'
)

train_loader, val_loader, test_loader = get_data_loaders(
    img_dir = img_dir,
    excel_path = excel_path,
    batch_size = batch_size,
    img_size = img_size,
    random_seed = seed,
    views = views,
    spec_preprocess_steps = spec_preprocess
)

# 优化器和损失函数
lr = 3e-4
weight_decay = 0

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

criterion = nn.SmoothL1Loss(beta=0.1)
# criterion = nn.MSELoss()

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

# 创建训练配置
config = TrainingConfig(
    device = 'cuda',
    epochs = 100
)

comment = f"model_config: f{model_config}, views: {views}, batch_size: {batch_size}, img_size: {img_size}, spec_preprocess: {spec_preprocess}, optimizer: AdamW, lr: {lr}, weight_decay: {weight_decay}, criterion: SmoothL1Loss, scheduler: ReduceLROnPlateau"

# 创建回调
callbacks = []
callbacks.append(ModelCheckpointCallback('models', 10, True))
callbacks.append(EarlyStoppingCallback(100))
callbacks.append(TensorBoardCallback('logs'))

# 创建训练器
trainer = Trainer(        
                    config,
                    model,
                    train_loader,
                    val_loader,
                    test_loader,
                    criterion,
                    optimizer,
                    scheduler,
                    callbacks,
                    comment=comment
                )

# 开始训练
print('开始训练...')
trainer.train()
print('训练完成!')