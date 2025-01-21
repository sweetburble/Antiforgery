import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, out_dim=256):
        super(CNNEncoder, self).__init__()
        # 예시로 간단히 4층 Conv
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),  # H/2
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # H/4
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # H/8
            nn.ReLU(),
            nn.Conv2d(128, 128, 4, stride=2, padding=1), # H/16
            nn.ReLU(),
        )
        self.fc = nn.Linear(128 * (16 * 16), out_dim)  
        # ↑ 만약 입력이 3×256×256이면, 4번 stride=2 하여 H,W를 16×16까지 줄이고 채널 128

    def forward(self, x):
        """
        x: [B, 3, H, W], LAB 이미지 (또는 RGB라도 동일), 값 범위는 대략 [-128,128] or 기타
        """
        # Clamp or rescale 등은 필요에 따라 추가
        out = self.conv_net(x)               # [B, 128, H/16, W/16]
        out = out.view(out.size(0), -1)      # Flatten
        out = self.fc(out)                   # [B, out_dim]
        return out


class ActorCriticCNN(nn.Module):
    def __init__(self, action_dim, encoder_out_dim=256):
        super(ActorCriticCNN, self).__init__()
        # CNN 인코더
        self.encoder = CNNEncoder(in_channels=3, out_dim=encoder_out_dim)
        
        # shared FC
        self.shared = nn.Sequential(
            nn.Linear(encoder_out_dim, 256),
            nn.ReLU()
        )
        # policy/value head
        self.policy = nn.Linear(256, action_dim)
        self.value  = nn.Linear(256, 1)

    def forward(self, x_lab):
        """
        x_lab: [B, 3, H, W], LAB (혹은 RGB)
        """
        # 1) CNN으로 임베딩 추출
        z = self.encoder(x_lab)   # [B, encoder_out_dim]
        # 2) shared
        h = self.shared(z)        # [B, 256]
        # 3) policy/value
        policy_logits = self.policy(h)
        value        = self.value(h).squeeze(-1)  # [B] 형태
        return policy_logits, value
