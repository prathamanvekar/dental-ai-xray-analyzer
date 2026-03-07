import torch
import torch.nn as nn
import timm

class CNNBiLSTM(nn.Module):
    """
    CNN-BiLSTM model architecture.
    Corrected to dynamically detect feature dimensions matching your trained weights.
    """

    def __init__(self, num_classes, backbone='efficientnet_b0',
                 lstm_hidden=256, lstm_layers=2, dropout=0.5):
        super(CNNBiLSTM, self).__init__()

        # CNN Backbone (Feature Extractor)
        self.backbone = timm.create_model(
            backbone,
            pretrained=False,
            features_only=True,
            out_indices=[-1]
        )

        # --- FIX START: Dynamically calculate feature dimension ---
        # This ensures it matches your trained weights (320 instead of 1280)
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)[-1]
            self.feature_dim = features.shape[1]
            # print(f"Detected backbone feature dim: {self.feature_dim}")
        # --- FIX END ---

        self.lstm_hidden = lstm_hidden

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.Tanh(),
            nn.Linear(lstm_hidden, 1),
            nn.Softmax(dim=1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, num_classes)
        )

    def forward(self, x):
        # CNN
        features = self.backbone(x)[-1]
        batch_size = features.size(0)

        # Reshape for LSTM [B, C, H, W] -> [B, H*W, C]
        features = features.view(batch_size, self.feature_dim, -1)
        features = features.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(features)

        # Attention
        att_weights = self.attention(lstm_out)
        context = torch.sum(att_weights * lstm_out, dim=1)

        # Classify
        output = self.classifier(context)
        return output
