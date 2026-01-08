from typing import Optional, Literal, Dict, Any
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class BasicBlock1d(nn.Module):
    """1D卷积基础块"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet1dEncoder(nn.Module):
    def __init__(
        self,
        block,  
        layers,
        in_channels=1,
        output_dim=None, 
        pool_type="avg",
        zero_init_residual=False,
    ):
        super().__init__()
        
        # 初始层
        self.in_channels = 64
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # 残差层
        self.layers = nn.ModuleList()
        channels = [64 * (2**i) for i in range(len(layers))]
        strides = [1] + [2] * (len(layers) - 1)
        
        for i, (channel, num_blocks, stride) in enumerate(zip(channels, layers, strides)):
            self.layers.append(self._make_layer(block, channel, num_blocks, stride))

        # 池化方式
        self.pool_type = pool_type
        if pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:  # None
            self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 投影层
        default_output_dim = channels[-1] * block.expansion
        if output_dim and output_dim != default_output_dim:
            self.proj = nn.Linear(default_output_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.proj = nn.Identity()
            self.output_dim = default_output_dim

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        layers.extend([block(self.in_channels, out_channels) for _ in range(1, blocks)])
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.layers:
            x = layer(x)

        if isinstance(self.pool, nn.Identity):
            features = x.view(x.size(0), -1)  # [B, C, L] -> [B, L, C]
        else:
            x = self.pool(x)  # [B, C, 1]
            features = x.view(x.size(0), -1)  # [B, C]
        
        return self.proj(features)  # 投影到目标维度

class VitEncoder(nn.Module):
    """ViT图像编码器"""
    def __init__(self, model_name='vit_tiny_patch16_224', pretrained=True, freeze_layers=False, output_dim=None):
        super(VitEncoder, self).__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        self.backbone.reset_classifier(num_classes=0)
        
        # 冻结参数
        if freeze_layers:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 投影层
        default_output_dim = self.backbone.num_features
        if output_dim and output_dim != default_output_dim:
            self.proj = nn.Linear(default_output_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.proj = nn.Identity()
            self.output_dim = default_output_dim

    def forward(self, x):
        features = self.backbone(x)
        return self.proj(features)
    
class TransformerEncoder(nn.Module):
    def __init__(self, 
                 input_channels=1, 
                 seq_len=100, 
                 embed_dim=64,
                 output_dim=None,
                 n_heads=4,             
                 n_layers=3, 
                 expansion_ratio=4, 
                 dropout=0.1,
                 use_cnn_preproc=True,
                 pool_type='mean'):
        super().__init__()
        
        # CNN预处理
        if use_cnn_preproc:
            self.pre_net = nn.Sequential(
                nn.Conv1d(input_channels, embed_dim//2, kernel_size=5, padding=2),
                nn.BatchNorm1d(embed_dim//2),
                nn.GELU(),
                nn.Conv1d(embed_dim//2, embed_dim, kernel_size=3, stride=2, padding=1),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            )
            seq_len = seq_len // 4 
        else:
            self.pre_net = nn.Linear(input_channels, embed_dim)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim*expansion_ratio,
            dropout=dropout,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 输出处理
        self.pool_type = pool_type
        if pool_type == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            
        self.output_dim = output_dim if output_dim is not None else embed_dim
        if self.output_dim != embed_dim:
            self.proj = nn.Linear(embed_dim, self.output_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        # 预处理
        if hasattr(self, 'pre_net') and isinstance(self.pre_net[0], nn.Conv1d):
            x = self.pre_net(x)          # [B,C,L] -> [B,D,L//4]
            x = x.permute(0, 2, 1)       # [B,D,L//4] -> [B,L//4,D]
        else:
            x = x.permute(0, 2, 1)       # [B,C,L] -> [B,L,C]
            x = self.pre_net(x)           # [B,L,C] -> [B,L,D]
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # 处理CLS Token
        if self.pool_type == 'cls':
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Transformer编码
        x = self.encoder(x)  # [B,L,D]
        
        # 输出处理
        if self.pool_type == 'cls':
            x = x[:, 0]                  # [B,D]
        elif self.pool_type == 'mean':
            x = x.mean(dim=1)             # [B,D]
        elif self.pool_type == 'max':
            x = x.max(dim=1).values       # [B,D]
        
        # 投影到目标维度 
        return self.proj(x)

class ResNetEncoder(nn.Module):
    def __init__(
        self,
        model_name="resnet18",
        pretrained=True,
        freeze_layers=False,
        output_dim=None, 
        pool_type="avg",
    ):
        super(ResNetEncoder, self).__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=pool_type,
        )

        # 冻结参数
        if freeze_layers:
            for param in self.backbone.parameters():
                param.requires_grad = False

        default_output_dim = self.backbone.num_features
        
        # 投影层
        if output_dim and output_dim != default_output_dim:
            self.proj = nn.Linear(default_output_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.proj = nn.Identity()
            self.output_dim = default_output_dim

    def forward(self, x):
        features = self.backbone(x)  # [batch_size, num_features]
        return self.proj(features)

class MultiViewEncoder(nn.Module):
    def __init__(
        self,
        base_encoder: nn.Module,
        num_views: int = 4,
        fusion_method: str = 'attention',
        output_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()

        assert hasattr(base_encoder, 'output_dim'), "Base encoder must have 'output_dim' attribute"
        
        self.views_encoder = nn.ModuleList([deepcopy(base_encoder) for _ in range(num_views)])
        self.fusion_method = fusion_method
        self.output_dim = output_dim if output_dim is not None else base_encoder.output_dim
        
        # 多视角融合策略
        if fusion_method == 'attention':
            self.view_attention = nn.Sequential(
                nn.Linear(base_encoder.output_dim, base_encoder.output_dim // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(base_encoder.output_dim // 2, 1),
                nn.Softmax(dim=1)
            )
            # 添加残差连接
            self.proj = nn.Linear(base_encoder.output_dim, self.output_dim) if output_dim else nn.Identity()
            
        elif fusion_method == 'lstm':
            self.lstm = nn.LSTM(
                input_size=base_encoder.output_dim,
                hidden_size=self.output_dim,
                num_layers=1,
                batch_first=True,
                dropout=dropout if dropout > 0 else 0
            )
            self.lstm_norm = nn.LayerNorm(self.output_dim)
            
        elif fusion_method == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=base_encoder.output_dim,
                nhead=4,
                dim_feedforward=2 * base_encoder.output_dim,
                dropout=dropout,
                batch_first=True
            )
            self.view_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.proj = nn.Linear(base_encoder.output_dim, self.output_dim)

    def forward(self, multi_view_images: torch.Tensor) -> torch.Tensor:

        batch_size, num_views = multi_view_images.shape[:2]
        
        # 各视角独立编码
        view_features = []
        for i in range(num_views):
            feat = self.views_encoder[i](multi_view_images[:, i])  # [B, D]
            view_features.append(feat)
        
        # 多视角融合
        all_feats = torch.stack(view_features, dim=1)  # [B, N, D]
        
        if self.fusion_method == 'mean':
            fused = all_feats.mean(dim=1)
        elif self.fusion_method == 'max':
            fused = all_feats.max(dim=1).values
        elif self.fusion_method == 'attention':
            weights = self.view_attention(all_feats)  # [B, N, 1]
            fused = (all_feats * weights).sum(dim=1)
            fused = self.proj(fused)
        elif self.fusion_method == 'lstm':
            _, (hidden, _) = self.lstm(all_feats)
            fused = self.lstm_norm(hidden.squeeze(0))
        elif self.fusion_method == 'transformer':
            fused = self.view_transformer(all_feats).mean(dim=1)
            fused = self.proj(fused)
            
        return fused
    
class CrossModalFusion(nn.Module):
    def __init__(
        self,
        in_dim_spectral: int,
        in_dim_image: int,
        method: str = 'concat',
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.method = method
        self.hidden_dim = hidden_dim
        
        if method == 'concat':
            # 拼接后投影
            self.proj = nn.Sequential(
                nn.Linear(in_dim_spectral + in_dim_image, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
        elif method == 'bilinear':
            # 双线性交互
            self.bilinear = nn.Bilinear(in_dim_spectral, in_dim_image, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(dropout)
            
        elif method == 'cross_attention':
            # 跨模态注意力
            assert in_dim_spectral == in_dim_image, "Cross-attention需要相同输入维度"
            self.query = nn.Linear(in_dim_spectral, hidden_dim)
            self.key = nn.Linear(in_dim_image, hidden_dim)
            self.value = nn.Linear(in_dim_image, hidden_dim)
            self.multihead_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(dropout)
            
        else:
            raise ValueError(f"Unknown fusion method: {method}")

    def forward(self, spectral_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:

        if self.method == 'concat':
            # print(spectral_feat.shape, image_feat.shape)
            fused = torch.cat([spectral_feat, image_feat], dim=-1)
            return self.proj(fused)
            
        elif self.method == 'bilinear':
            fused = self.bilinear(spectral_feat, image_feat)
            return self.dropout(self.norm(F.relu(fused)))
            
        elif self.method == 'cross_attention':
            # 使用光谱特征作为query，图像特征作为key/value
            query = self.query(spectral_feat).unsqueeze(1)  # [B, 1, D]
            key = self.key(image_feat).unsqueeze(1)
            value = self.value(image_feat).unsqueeze(1)
            
            attn_output, _ = self.multihead_attn(
                query=query,
                key=key,
                value=value,
                need_weights=False
            )
            fused = self.dropout(self.norm(attn_output.squeeze(1)))
            return fused

    
class AppleSugarModel(nn.Module):
    def __init__(
        self,
        spectral_encoder: Optional[nn.Module] = None,
        image_encoder: Optional[nn.Module] = None,
        fusion_method: str = 'concat',
        hidden_dim: int = 512,
        output_dim: int = 1,
        dropout: float = 0.3,
        output_activation: Optional[str] = None
    ):
        super().__init__()
        
        # 编码器检查
        assert spectral_encoder or image_encoder, "At least one encoder needs to be provided"
        self.spectral_encoder = spectral_encoder
        self.image_encoder = image_encoder
        self.is_multimodal = bool(spectral_encoder and image_encoder)
        
        # 单模态输出
        if not self.is_multimodal:
            encoder = spectral_encoder if spectral_encoder else image_encoder
            self.proj = nn.Sequential(
                nn.Linear(encoder.output_dim, output_dim),
            )
        
        # 多模态融合
        else:
            self.fusion = CrossModalFusion(
                in_dim_spectral=spectral_encoder.output_dim,
                in_dim_image=image_encoder.output_dim,
                method=fusion_method,
                hidden_dim=hidden_dim
            )
            self.output = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim//2, output_dim)
            )
        
        # 输出激活
        self.output_act = self._get_activation(output_activation)

    def forward(self, 
               spectral: Optional[torch.Tensor] = None, 
               images: Optional[torch.Tensor] = None):

        assert spectral is not None or images is not None, "At least one input is required"
        images = images[:, 0, :, :, :]  # 单视角

        if not self.is_multimodal:
            if self.spectral_encoder:
                features = self.spectral_encoder(spectral)
            else:
                features = self.image_encoder(images)

            output = self.proj(features)
        
        else:
            assert spectral is not None and images is not None, "Dual-mode requires two inputs"
            spectral_feat = self.spectral_encoder(spectral)
            # images = images[:, 0, :, :]
            image_feat = self.image_encoder(images)
            fused = self.fusion(spectral_feat, image_feat)
            output = self.output(fused)
        
        return self.output_act(output.squeeze(-1)) if self.output_act else output.squeeze(-1)

    def _get_activation(self, name):
        return {
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            None: None
        }.get(name)

def get_spectral_encoder(config: Dict[str, Any]) -> nn.Module:
    if config['type'] == 'resnet1d':
        return ResNet1dEncoder(
            block=BasicBlock1d,
            in_channels=config.get('in_channels', 1),
            output_dim=config.get('output_dim', None),
            layers=config.get('layers', [2,2]),
            pool_type=config.get('pool_type', 'avg'),
            zero_init_residual=config.get('zero_init_residual', False)
        )
        
    elif config['type'] == 'transformer':
        return TransformerEncoder(
            input_channels=config.get('in_channels', 1),
            seq_len=config.get('seq_len', 100),
            embed_dim=config.get('embed_dim', 64),
            output_dim=config.get('output_dim', None),
            n_heads=config.get('n_heads', 4),
            n_layers=config.get('n_layers', 3),
            expansion_ratio=config.get('expansion_ratio', 4),
            dropout=config.get('dropout', 0.1),
            use_cnn_preproc=config.get('use_cnn_preproc', True),
            pool_type=config.get('pool_type', 'mean')
        )
    else:
        raise ValueError(f"Unknown spectral encoder type: {config['type']}")

def get_image_encoder(config: Dict[str, Any]) -> nn.Module:
    if config['type'] in ['resnet', 'vit']:
        return get_base_image_encoder(config)
        
    if config['type'] == 'multiview':
        base_encoder_config = config.get('base_encoder', None)
        if not base_encoder_config:
            raise ValueError("Multiview encoder requires 'base_encoder' configuration")
        base_encoder = get_base_image_encoder(base_encoder_config)
        return MultiViewEncoder(
            base_encoder=base_encoder,
            num_views=config.get('num_views', 2),
            fusion_method=config.get('fusion_method', 'attention'),
            output_dim=config.get('output_dim', None),
            dropout=config.get('dropout', 0.1)
        )
        
def get_base_image_encoder(config: Dict[str, Any]) -> nn.Module:
    if config['type'] == 'resnet':
        return ResNetEncoder(
            model_name=config.get('model_name', 'resnet18'),
            pretrained=config.get('pretrained', True),
            freeze_layers=config.get('freeze_layers', False),
            output_dim=config.get('output_dim', None),
            pool_type=config.get('pool_type', 'avg')
        )
    elif config['type'] == 'vit':
        return VitEncoder(
            model_name=config.get('model_name', 'vit_tiny_patch16_224'),
            pretrained=config.get('pretrained', True),
            freeze_layers=config.get('freeze_layers', False),
            output_dim=config.get('output_dim', None)
        )
    else:
        raise ValueError(f"Unknown image encoder type: {config['type']}")

def get_model(
    # 编码器配置
    encoder_config: Dict[str, Any],
    # 通用配置
    fusion_method: str = 'concat',
    hidden_dim: int = 512,
    output_dim: int = 1,
    dropout: float = 0.3,
    output_activation: Optional[str] = None
) -> AppleSugarModel:
    
    if 'spectral_encoder' in encoder_config:
        spectral_encoder = get_spectral_encoder(encoder_config['spectral_encoder'])
    else:
        spectral_encoder = None
    
    if 'image_encoder' in encoder_config:
        image_encoder = get_image_encoder(encoder_config['image_encoder'])
    else:
        image_encoder = None
    
    return AppleSugarModel(
        spectral_encoder=spectral_encoder,
        image_encoder=image_encoder,
        fusion_method=fusion_method,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
        output_activation=output_activation
    )