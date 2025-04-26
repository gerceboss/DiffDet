import torch
import torch.nn as nn

class DetectionDecoder(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=512):
        super(DetectionDecoder, self).__init__()
        self.fc1 = nn.Linear(4 + hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.box_head = nn.Linear(hidden_dim, 4)    # predict box offsets
        self.cls_head = nn.Linear(hidden_dim, num_classes)  # predict class
        
    def forward(self, noisy_boxes, features):
        """
        Args:
            noisy_boxes: (B, N, 4)
            features: (B, C, H, W) - image features
        """
        B, N, _ = noisy_boxes.shape
        # Simple: average pool feature map
        pooled_features = features.mean(dim=[2,3])  # (B, C)

        # expand to match boxes
        pooled_features = pooled_features.unsqueeze(1).expand(-1, N, -1)  # (B, N, C)
        x = torch.cat([pooled_features, noisy_boxes], dim=-1)  # (B, N, C+4)
        
        # Reshape for linear layer: (B*N, C+4)
        x = x.reshape(-1, x.size(-1))
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        
        # Reshape back to (B, N, hidden_dim)
        x = x.reshape(B, N, -1)
        
        # Predict boxes and classes
        pred_boxes = self.box_head(x)  # (B, N, 4)
        pred_logits = self.cls_head(x)  # (B, N, num_classes)
        
        return pred_boxes, pred_logits
