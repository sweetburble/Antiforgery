import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torchvision import transforms as T
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
    model.load_state_dict(pretrained_dict, strict=False)


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


# LAB Attack
def lab_attack(X_nat, c_trg, model, epsilon=0.05, iter=100, gamma=0.99, clip_range=0.2, epochs=4):
    """LAB 공간에서의 공격: Actor-Critic 및 PPO 적용"""
    criterion = nn.MSELoss().cuda()

    # Perturbation 초기화
    pert_a = torch.zeros(X_nat.shape[0], 2, X_nat.shape[2], X_nat.shape[3]).cuda().requires_grad_()
    optimizer = torch.optim.Adam([pert_a], lr=1e-4, betas=(0.9, 0.999))

    # RL 모델 초기화
    state_dim = X_nat.numel()
    action_dim = 2  # Increase/Decrease perturbation
    rl_model = ActorCritic(state_dim, action_dim).cuda()
    rl_optimizer = torch.optim.Adam(rl_model.parameters(), lr=1e-4)

    X = denorm(X_nat.clone())

    for step in range(iter):
        # RGB 클램핑
        X = torch.clamp(X, min=0, max=1)
        X_lab = rgb2lab(X).cuda()

        # LAB 클램핑
        X_lab = torch.clamp(X_lab, min=-128, max=128)

        state = X_lab.view(-1).detach()

        # NaN 및 Inf 디버깅
        if torch.isnan(X).any() or torch.isinf(X).any():
            print(f"[Error] NaN or Inf detected in X at step {step}")
            print(f"X min: {X.min()}, X max: {X.max()}")
            break

        if torch.isnan(X_lab).any() or torch.isinf(X_lab).any():
            print(f"[Error] NaN or Inf detected in X_lab at step {step}")
            print(f"X_lab min: {X_lab.min()}, X_lab max: {X_lab.max()}")
            break

        if torch.isnan(state).any() or torch.isinf(state).any():
            print(f"[Error] NaN or Inf detected in state at step {step}")
            print(f"state min: {state.min()}, state max: {state.max()}")
            break

        # Actor-Critic: 행동 결정
        policy, value = rl_model(state)

        # 정책 점검 및 정규화
        if torch.isnan(policy).any() or torch.isinf(policy).any():
            print(f"[Error] NaN or Inf detected in policy at step {step}")
            break

        policy = F.softmax(policy, dim=-1)

        try:
            action = torch.multinomial(policy, 1).item()
        except RuntimeError as e:
            print(f"[Error] torch.multinomial failed at step {step} with policy: {policy}")
            raise

        
        perturb_change = epsilon if action == 1 else -epsilon
        pert_a.data += perturb_change

        # Perturbation 적용 및 클램핑
        X_lab[:, 1:, :, :] = torch.clamp(X_lab[:, 1:, :, :] + pert_a, min=-128, max=128)

        try:
            X_new = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(lab2rgb(X_lab))
        except Exception as e:
            print(f"Error in LAB to RGB conversion: {e}")
            break

        gen_noattack, _ = model(X_nat, c_trg[step % len(c_trg)])
        gen_attack, _ = model(X_new, c_trg[step % len(c_trg)])

        # PSNR 및 SSIM 계산
        psnr_input_gan, ssim_input_gan = compare(X_nat, X_new) # 원본 이미지와 섭동이 추가된 방어 이미지 비교

        psnr_attack_gan, ssim_attack_gan = compare(gen_noattack, gen_attack) # 원본 이미지/방어 이미지에 대한 GAN 생성 이미지들을 비교

        # 보상 계산 (Attack-GAN의 PSNR 및 SSIM을 최소화하고, Input-GAN의 PSNR 및 SSIM을 최대화)
        reward = - (psnr_attack_gan + ssim_attack_gan) + (psnr_input_gan + ssim_input_gan)

        # Loss 및 디버깅 출력
        print(f"[Iteration {step}] Loss: {reward:.4f}, "
              f"PSNR Input-GAN: {psnr_input_gan:.4f}, SSIM Input-GAN: {ssim_input_gan:.4f}, "
              f"PSNR Attack-GAN: {psnr_attack_gan:.4f}, SSIM Attack-GAN: {ssim_attack_gan:.4f}")

        # PPO 최적화
        old_policy = policy.detach()
        _, next_value = rl_model(state)

        for epoch in range(epochs):
            policy, value = rl_model(state)

            if len(policy.shape) == 1:
                policy = policy.unsqueeze(0)

            action_tensor = torch.tensor([[action]]).cuda()

            policy_loss = -torch.min(
                torch.clamp(policy.gather(1, action_tensor), 1 - clip_range, 1 + clip_range),
                reward + gamma * next_value.detach()
            )
            value_loss = F.mse_loss(value, reward + gamma * next_value.detach())

            rl_optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            rl_optimizer.step()

        # Perturbation 업데이트
        optimizer.zero_grad()
        criterion(gen_attack, gen_noattack).backward()
        optimizer.step()

        X = X_new.detach()

    return X, X - X_nat