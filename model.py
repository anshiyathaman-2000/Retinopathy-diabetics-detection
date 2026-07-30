import torch
import torch.nn as nn
import torchvision.models as models

class HybridCNNTransformerLSTM(nn.Module):
    def __init__(self, num_classes=46, d_model=256, nhead=4, num_transformer_layers=2, dim_feedforward=1024, dropout=0.1):
        """
        CNN-Transformer-LSTM Hybrid Architecture for Fundus Image Multi-Label Classification.
        
        Args:
            num_classes (int): Number of target disease categories (46).
            d_model (int): Hidden dimension size for Transformer and LSTM layers.
            nhead (int): Number of attention heads in the Transformer Encoder.
            num_transformer_layers (int): Number of Transformer Encoder layers.
            dim_feedforward (int): Feedforward network dimension in Transformer.
            dropout (float): Dropout probability.
        """
        super(HybridCNNTransformerLSTM, self).__init__()
        
        # 1. CNN Feature Extractor (ResNet50 Backbone)
        # Using pre-trained weights from ImageNet
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Extract layers up to layer4
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
        
        # Freeze early layers of CNN to preserve pre-trained features and speed up training
        # We will keep layer3 and layer4 unfrozen for fine-tuning
        for param in self.cnn_backbone[:-2].parameters():
            param.requires_grad = False
            
        # 2. Linear Projection Layer
        # Reduces the 2048 CNN channels to d_model (256)
        self.proj = nn.Conv2d(2048, d_model, kernel_size=1)
        
        # 3. Position Encoding
        # Since spatial size of layer4 output is 8x8 = 64 patches, we learn 64 position embeddings
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
        # Input: [B, 64, d_model] -> Output: [B, 64, d_model]
        # Since it is bidirectional, we set hidden_size = d_model // 2
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 6. Multi-Label Classification Head
        # Standard dropout and fully connected layer to output logits
        self.fc_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x shape: [B, 3, H, W] -> e.g., [B, 3, 256, 256]
        
        # Step 1: CNN Feature Extraction
        cnn_features = self.cnn_backbone(x)  # shape: [B, 2048, H/32, W/32] -> [B, 2048, 8, 8]
        
        # Step 2: Channel Projection
        projected = self.proj(cnn_features)  # shape: [B, d_model, 8, 8] -> [B, 256, 8, 8]
        
        # Step 3: Flatten Spatial Grid to Sequence of Tokens
        # Flatten [B, C, H, W] to [B, C, H*W] and transpose to [B, H*W, C] (Sequence format)
        B, C, H, W = projected.shape
        seq_len = H * W
        features_seq = projected.view(B, C, seq_len).transpose(1, 2)  # shape: [B, 64, 256]
        
        # Step 4: Add Position Embeddings
        features_seq = features_seq + self.pos_embedding
        
        # Step 5: Global Context modeling via Transformer
        trans_out = self.transformer_encoder(features_seq)  # shape: [B, 64, 256]
        
        # Step 6: Spatial Sequence modeling via Bidirectional LSTM
        lstm_out, _ = self.lstm(trans_out)  # shape: [B, 64, 256]
        
        # Step 7: Global Average Pooling over the patch sequence
        pooled = torch.mean(lstm_out, dim=1)  # shape: [B, 256]
        
        # Step 8: Multi-Label Logits
        logits = self.classifier(self.fc_dropout(pooled))  # shape: [B, num_classes] -> [B, 46]
        
        return logits

if __name__ == "__main__":
    # Test script for Hybrid model
    print("Initializing hybrid model...")
    model = HybridCNNTransformerLSTM(num_classes=46)
    
    # Create a dummy batch of 2 fundus images
    dummy_input = torch.randn(2, 3, 256, 256)
    print(f"Input batch shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits = model(dummy_input)
        
    print(f"Forward pass completed. Output shape: {logits.shape}")
    assert logits.shape == (2, 46), "Shape mismatch!"
    print("Success! Model works as expected.")
