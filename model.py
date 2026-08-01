import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class HybridCNNTransformerLSTM(nn.Module):
    def __init__(self, num_classes=46, d_model=256, nhead=4, num_transformer_layers=2, dim_feedforward=1024, dropout=0.1):
        """
        CNN-Transformer-LSTM Hybrid Architecture for Fundus Image Multi-Label Classification.
        """
        super(HybridCNNTransformerLSTM, self).__init__()
        
        # 1. CNN Feature Extractor (ResNet50 Backbone)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        self.cnn_backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # [B, 256, H/4, W/4]
            resnet.layer2,  # [B, 512, H/8, W/8]
            resnet.layer3,  # [B, 1024, H/16, W/16]
            resnet.layer4   # [B, 2048, H/32, W/32] -> for 256x256 input, output shape is [B, 2048, 8, 8]
        )
        
        # Freeze early layers
        for param in self.cnn_backbone[:-2].parameters():
            param.requires_grad = False
            
        # 2. Linear Projection Layer
        self.proj = nn.Conv2d(2048, d_model, kernel_size=1)
        
        # 3. Position Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        
        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        
        # 5. Spatial Sequence Layer (Bidirectional LSTM)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 6. Multi-Label Classification Head
        self.fc_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        cnn_features = self.cnn_backbone(x)
        projected = self.proj(cnn_features)
        
        B, C, H, W = projected.shape
        seq_len = H * W
        features_seq = projected.view(B, C, seq_len).transpose(1, 2)
        
        features_seq = features_seq + self.pos_embedding
        
        trans_out = self.transformer_encoder(features_seq)
        
        lstm_out, _ = self.lstm(trans_out)
        
        pooled = torch.mean(lstm_out, dim=1)
        
        logits = self.classifier(self.fc_dropout(pooled))
        return logits


class ResNet50Baseline(nn.Module):
    def __init__(self, num_classes=46, dropout=0.1):
        """
        Standalone ResNet50 Baseline for Fundus Image Classification.
        """
        super(ResNet50Baseline, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # Freeze early layers
        for param in self.backbone[:-2].parameters():
            param.requires_grad = False
            
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        pooled = self.avgpool(features).flatten(1)
        logits = self.fc(pooled)
        return logits


class ViTBaseline(nn.Module):
    def __init__(self, num_classes=46, patch_size=32, d_model=256, nhead=4, num_layers=4, dim_feedforward=1024, dropout=0.1):
        """
        Custom Vision Transformer Baseline operating on patch embeddings.
        """
        super(ViTBaseline, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Patch projection to project non-overlapping patches
        self.patch_proj = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        
        # For 256x256 image with patch_size=32, we have (256/32)^2 = 64 tokens
        num_patches = (256 // patch_size) * (256 // patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.fc_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x: [B, 3, 256, 256]
        projected = self.patch_proj(x)  # [B, d_model, 8, 8]
        B, C, H, W = projected.shape
        seq_len = H * W
        
        features_seq = projected.view(B, C, seq_len).transpose(1, 2)  # [B, 64, d_model]
        features_seq = features_seq + self.pos_embedding
        
        trans_out = self.transformer_encoder(features_seq)  # [B, 64, d_model]
        
        pooled = torch.mean(trans_out, dim=1)  # [B, d_model]
        logits = self.classifier(self.fc_dropout(pooled))
        return logits


class CNNTransformerHybrid(nn.Module):
    def __init__(self, num_classes=46, d_model=256, nhead=4, num_transformer_layers=2, dim_feedforward=1024, dropout=0.1):
        """
        CNN + Transformer Encoder Hybrid (No LSTM sequential layer).
        """
        super(CNNTransformerHybrid, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        self.cnn_backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # Freeze early layers
        for param in self.cnn_backbone[:-2].parameters():
            param.requires_grad = False
            
        self.proj = nn.Conv2d(2048, d_model, kernel_size=1)
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        
        self.fc_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        cnn_features = self.cnn_backbone(x)
        projected = self.proj(cnn_features)
        
        B, C, H, W = projected.shape
        seq_len = H * W
        features_seq = projected.view(B, C, seq_len).transpose(1, 2)
        
        features_seq = features_seq + self.pos_embedding
        
        trans_out = self.transformer_encoder(features_seq)
        
        pooled = torch.mean(trans_out, dim=1)
        logits = self.classifier(self.fc_dropout(pooled))
        return logits


class CNNLSTMHybrid(nn.Module):
    def __init__(self, num_classes=46, d_model=256, dropout=0.1):
        """
        CNN + LSTM Hybrid (No Transformer global attention).
        """
        super(CNNLSTMHybrid, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        self.cnn_backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # Freeze early layers
        for param in self.cnn_backbone[:-2].parameters():
            param.requires_grad = False
            
        self.proj = nn.Conv2d(2048, d_model, kernel_size=1)
        
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        cnn_features = self.cnn_backbone(x)
        projected = self.proj(cnn_features)
        
        B, C, H, W = projected.shape
        seq_len = H * W
        features_seq = projected.view(B, C, seq_len).transpose(1, 2)
        
        lstm_out, _ = self.lstm(features_seq)
        
        pooled = torch.mean(lstm_out, dim=1)
        logits = self.classifier(self.fc_dropout(pooled))
        return logits


class CNNInceptionHybrid(nn.Module):
    def __init__(self, num_classes=46, dropout=0.1):
        """
        Dual-Stream CNN backbone combining ResNet50 and InceptionV3 features.
        """
        super(CNNInceptionHybrid, self).__init__()
        # ResNet50 Stream
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet_backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        for param in self.resnet_backbone[:-2].parameters():
            param.requires_grad = False
            
        # InceptionV3 Stream (aux_logits defaults to True, which is required for loading default weights)
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.inception_backbone = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            inception.maxpool1,
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            inception.maxpool2,
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c
        )
        
        # Freeze Inception early layers (up to Mixed_6a)
        freeze = True
        for name, child in self.inception_backbone.named_children():
            if name == "Mixed_6a":
                freeze = False
            if freeze:
                for param in child.parameters():
                    param.requires_grad = False
                    
        self.resnet_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.inception_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fused classifier
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048 + 2048, num_classes)
        )
        
    def forward(self, x):
        # ResNet path
        res_features = self.resnet_backbone(x)
        res_pooled = self.resnet_pool(res_features).flatten(1)
        
        # InceptionV3 path requires 299x299 size
        x_inc = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        inc_features = self.inception_backbone(x_inc)
        inc_pooled = self.inception_pool(inc_features).flatten(1)
        
        # Concat features
        fused = torch.cat([res_pooled, inc_pooled], dim=1)
        logits = self.fc(fused)
        return logits


def get_model(model_name, num_classes=46, **kwargs):
    """
    Factory function to instantiate models dynamically.
    """
    m_name = model_name.lower().strip()
    if m_name in ["hybrid", "hybridcnntransformerlstm"]:
        return HybridCNNTransformerLSTM(num_classes=num_classes, **kwargs)
    elif m_name in ["resnet50", "resnet50baseline"]:
        return ResNet50Baseline(num_classes=num_classes, **kwargs)
    elif m_name in ["vit", "vitbaseline"]:
        return ViTBaseline(num_classes=num_classes, **kwargs)
    elif m_name in ["cnn_transformer", "cnn_transformer_hybrid"]:
        return CNNTransformerHybrid(num_classes=num_classes, **kwargs)
    elif m_name in ["cnn_lstm", "cnn_lstm_hybrid"]:
        return CNNLSTMHybrid(num_classes=num_classes, **kwargs)
    elif m_name in ["cnn_inception", "cnn_inception_hybrid"]:
        return CNNInceptionHybrid(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    print("Testing model library instantiations...")
    dummy_input = torch.randn(2, 3, 256, 256)
    
    models_to_test = ["hybrid", "resnet50", "vit", "cnn_transformer", "cnn_lstm", "cnn_inception"]
    for m in models_to_test:
        print(f"\nInitializing model: {m}...")
        model = get_model(m, num_classes=46)
        with torch.no_grad():
            logits = model(dummy_input)
        print(f"-> Output shape: {logits.shape}")
        assert logits.shape == (2, 46), f"Output shape mismatch for {m}!"
    print("\nAll models instantiated and verified successfully!")
