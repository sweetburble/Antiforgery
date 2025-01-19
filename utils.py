import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torchvision import transforms as T
from torch.distributions import Categorical
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize
from color_space import *  # RGB-LAB 변환 모듈


# Actor-Critic Model
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU()
        )
        self.policy = nn.Linear(256, action_dim)
        self.value = nn.Sequential(
            nn.Linear(256, 1)
        )

    def forward(self, state):
        x = self.shared(state)
        
        # Policy logits 계산 및 안정화
        policy_logits = self.policy(x)
        policy_logits = policy_logits - policy_logits.max()  # 안정화
        
        # Softmax로 확률 변환
        policy = F.softmax(policy_logits, dim=-1)
        
        # Value 계산
        value = self.value(x)
        
        return policy, value


# Load Model Weights
def load_model_weights(model, path):
    pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage, weights_only=True)
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'preprocessing' not in k}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)


# Denormalization
def denorm(x):
    """[-1, 1] 범위를 [0, 1]로 변환"""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


# Label to One-Hot Encoding
def label2onehot(labels, dim):
    """레이블 인덱스를 원-핫 벡터로 변환"""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out


# Create Target Labels
def create_labels(c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
    """타겟 도메인 레이블 생성"""
    if dataset == 'CelebA':
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)

    c_trg_list = []
    for i in range(c_dim):
        if dataset == 'CelebA':
            c_trg = c_org.clone()
            if i in hair_color_indices:
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)
        elif dataset == 'RaFD':
            c_trg = label2onehot(torch.ones(c_org.size(0)) * i, c_dim)

        c_trg_list.append(c_trg.cuda())
    return c_trg_list


# Compare Images (SSIM and PSNR)
def compare(img1, img2):
    """이미지 비교 (SSIM 및 PSNR 계산)"""
    # Detach tensors and convert to numpy arrays
    img1_np = img1.squeeze(0).detach().cpu().numpy()
    img2_np = img2.squeeze(0).detach().cpu().numpy()
    img1_np = np.transpose(img1_np, (1, 2, 0))
    img2_np = np.transpose(img2_np, (1, 2, 0))

    h, w, c = img1_np.shape
    min_size = 7
    if h < min_size or w < min_size:
        img1_np = resize(img1_np, (max(min_size, h), max(min_size, w)), anti_aliasing=True)
        img2_np = resize(img2_np, (max(min_size, h), max(min_size, w)), anti_aliasing=True)

    win_size = min(img1_np.shape[0], img1_np.shape[1], 7)

    try:
        ssim = structural_similarity(img1_np, img2_np, channel_axis=-1, win_size=win_size, data_range=1.0)
        psnr = peak_signal_noise_ratio(img1_np, img2_np, data_range=1.0)
    except ValueError as e:
        print(f"Error in SSIM calculation: {e}")
        raise
    return ssim, psnr


def lab_attack(X_nat, c_trg, model, epsilon=0.05, iter=100, gamma=0.99, 
               clip_range=0.2, epochs=4, patch_size=16):
    """
    PPO + 패치 단위 액션 + LAB 색공간 공격
    - Generator G는 eval 모드로 고정
    - a/b 채널만 Parameter로 최적화
    - PPO가 주도하는 구조
    """
    # 1. Generator G 고정
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # 2. 패치 관련 설정
    B, C, H, W = X_nat.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w
    
    # 액션 차원 = 각 패치마다 (+/-) 2개씩 → 2 × 패치 개수
    action_dim = 2 * num_patches
    # 상태 차원(기존 코드처럼, 이미지 전체를 flatten)
    state_dim  = X_nat.numel()

    # 3. LAB 변환 및 a/b 채널 파라미터화
    X = denorm(X_nat.clone()).clamp(0, 1)
    X_lab_full = rgb2lab(X).clamp(-128, 128)
    
    # a/b 채널만 Parameter로 관리
    pert_ab = nn.Parameter(X_lab_full[:, 1:, :, :].clone())
    pert_optimizer = torch.optim.Adam([pert_ab], lr=1e-4)

    # 4. RL 모델 설정
    rl_model = ActorCritic(state_dim, action_dim).cuda()
    rl_optimizer = torch.optim.Adam(rl_model.parameters(), lr=1e-4)

    for step in range(iter):
        # 5. 현재 상태 구성 (L 채널 고정 + a/b는 pert_ab)
        X_lab_full[:, 1:, :, :] = pert_ab
        state = X_lab_full.view(-1).detach()

        # 6. Actor-Critic으로 액션 선택
        policy_logits, value = rl_model(state)
        dist = Categorical(logits=policy_logits)
        action = dist.sample()
        old_log_prob = dist.log_prob(action).detach()
        old_value = value.detach()

        # 7. 패치 위치 계산 및 섭동 적용
        patch_idx = action.item() // 2
        sign_idx = action.item() % 2
        py = patch_idx // num_patches_w
        px = patch_idx % num_patches_w
        y0, y1 = py * patch_size, (py + 1) * patch_size
        x0, x1 = px * patch_size, (px + 1) * patch_size

        # 선택된 패치에만 섭동 적용
        with torch.no_grad():
            perturb = epsilon if sign_idx == 1 else -epsilon
            pert_ab.data[:, :, y0:y1, x0:x1] += perturb
            pert_ab.data.clamp_(-128, 128)

        # 8. LAB → RGB 변환
        X_lab_full[:, 1:, :, :] = pert_ab
        X_new_rgb = lab2rgb(X_lab_full.clamp(-128, 128))
        X_new = T.Normalize([0.5]*3, [0.5]*3)(X_new_rgb)

        # 9. Generator 추론
        gen_noattack, _ = model(X_nat, c_trg[step % len(c_trg)])
        gen_attack, _ = model(X_new, c_trg[step % len(c_trg)])

        # 10. 보상 계산
        ssim_input, psnr_input = compare(X_nat, X_new)
        ssim_attack, psnr_attack = compare(gen_noattack, gen_attack)
        reward = -(psnr_attack + ssim_attack) + (psnr_input + ssim_input)

        # 11. PPO 업데이트
        _, next_value = rl_model(state)
        advantage = reward + gamma * next_value.detach() - old_value

        for epoch in range(epochs):
            # Policy & Value 업데이트
            new_policy_logits, new_value = rl_model(state)
            dist_new = Categorical(logits=new_policy_logits)
            new_log_prob = dist_new.log_prob(action)
            
            ratio = torch.exp(new_log_prob - old_log_prob)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-clip_range, 1+clip_range) * advantage
            
            policy_loss = -torch.min(surr1, surr2)
            value_loss = F.mse_loss(new_value, advantage + old_value)
            
            rl_loss = policy_loss + value_loss
            rl_optimizer.zero_grad()
            rl_loss.backward()
            rl_optimizer.step()

    return X_new, X_new - X_nat
