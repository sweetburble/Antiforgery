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


def lab_attack(X_nat, c_trg, model, epsilon=0.05, iter=100, gamma=0.99, clip_range=0.2, epochs=4):
    """LAB 공간에서의 공격: Actor-Critic 및 PPO 적용"""
    criterion = nn.MSELoss().cuda()

    # (a, b) 채널에 대한 2채널짜리 perturbation
    pert_a = torch.zeros(X_nat.shape[0], 2, X_nat.shape[2], X_nat.shape[3]).cuda().requires_grad_()
    optimizer = torch.optim.Adam([pert_a], lr=1e-4, betas=(0.9, 0.999))

    # RL 모델 초기화
    state_dim = X_nat.numel()
    action_dim = 2  # Increase/Decrease perturbation
    rl_model = ActorCritic(state_dim, action_dim).cuda()
    rl_optimizer = torch.optim.Adam(rl_model.parameters(), lr=1e-4)

    # -1~1 범위를 [0,1]로 바꿔주고 시작
    X = denorm(X_nat.clone())

    for step in range(iter):
        # 1) RGB 영역 clamp [0, 1]
        X = torch.clamp(X, min=0.0, max=1.0)

        # 2) RGB -> LAB 변환
        X_lab = rgb2lab(X).cuda()

        # ※ 여기서 L, a, b 범위를 채널별로 따로 clamp
        #    L 채널: [0, 100], a/b 채널: [-128, 128]
        X_lab[:, 0, :, :] = torch.clamp(X_lab[:, 0, :, :], min=0.0, max=100.0)
        X_lab[:, 1, :, :] = torch.clamp(X_lab[:, 1, :, :], min=-128.0, max=128.0)
        X_lab[:, 2, :, :] = torch.clamp(X_lab[:, 2, :, :], min=-128.0, max=128.0)

        # 현재 state를 (LAB 전체를 1D로 펼친) 벡터로 사용
        state = X_lab.view(-1).detach()

        # NaN/Inf 체크
        if torch.isnan(X_lab).any() or torch.isinf(X_lab).any():
            print(f"[Error] NaN or Inf detected in X_lab at step {step}")
            break
        if torch.isnan(state).any() or torch.isinf(state).any():
            print(f"[Error] NaN or Inf detected in state at step {step}")
            break

        # Actor-Critic으로 액션 결정
        policy, value = rl_model(state)
        if torch.isnan(policy).any() or torch.isinf(policy).any():
            print(f"[Error] NaN or Inf detected in policy at step {step}")
            break

        policy = F.softmax(policy, dim=-1)
        try:
            action = torch.multinomial(policy, 1).item()
        except RuntimeError as e:
            print(f"[Error] torch.multinomial failed at step {step} with policy: {policy}")
            raise

        # action == 1이면 (+ epsilon), action == 0이면 (- epsilon)
        perturb_change = epsilon if action == 1 else -epsilon
        
        pert_a = pert_a.clone() + perturb_change

        # 실제 LAB 공간의 (a, b) 채널에 perturbation 적용
        X_lab[:, 1:, :, :] += pert_a

        # 다시 채널별 clamp 수행
        X_lab = X_lab.clone()
        X_lab[:, 0, :, :] = torch.clamp(X_lab[:, 0, :, :].clone(), min=0.0, max=100.0)
        X_lab[:, 1, :, :] = torch.clamp(X_lab[:, 1, :, :].clone(), min=-128.0, max=128.0)
        X_lab[:, 2, :, :] = torch.clamp(X_lab[:, 2, :, :].clone(), min=-128.0, max=128.0)

        try:
            # LAB -> RGB (sRGB) 변환 후, -1~1 범위를 쓰는 모델이라면 Normalize 적용
            X_new = T.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])(lab2rgb(X_lab))
        except Exception as e:
            print(f"Error in LAB to RGB conversion: {e}")
            break

        # 3) 모델에 넣어 GAN 생성
        gen_noattack, _ = model(X_nat, c_trg[step % len(c_trg)])
        gen_attack, _   = model(X_new, c_trg[step % len(c_trg)])

        # 4) PSNR/SSIM 계산
        psnr_input_gan, ssim_input_gan   = compare(X_nat, X_new)
        psnr_attack_gan, ssim_attack_gan = compare(gen_noattack, gen_attack)

        # 5) 보상 계산
        #   - (원본/섭동 GAN 비교) PSNR,SSIM을 '최대화' -> (+)
        #   - (딥페이크 깨뜨리기, 즉 Attack-GAN PSNR,SSIM을 '최소화') -> (-)
        reward = -(psnr_attack_gan + ssim_attack_gan) + (psnr_input_gan + ssim_input_gan)

        # 디버깅 용도
        print(f"[Iteration {step}] Reward: {reward:.4f}, "
              f"PSNR Input-GAN: {psnr_input_gan:.4f}, SSIM Input-GAN: {ssim_input_gan:.4f}, "
              f"PSNR Attack-GAN: {psnr_attack_gan:.4f}, SSIM Attack-GAN: {ssim_attack_gan:.4f}")

        # 6) PPO 업데이트 (간소화 형태)
        old_policy = policy.detach()
        _, next_value = rl_model(state)

        for epoch in range(epochs):
            new_policy, new_value = rl_model(state)
            if len(new_policy.shape) == 1:
                new_policy = new_policy.unsqueeze(0)

            action_tensor = torch.tensor([[action]]).cuda()
            # (주의) 실제로는 ratio 계산, advantage 등 추가가 필요
            policy_loss = -torch.min(
                torch.clamp(new_policy.gather(1, action_tensor), 1 - clip_range, 1 + clip_range),
                reward + gamma * next_value.detach()
            )
            value_loss = F.mse_loss(new_value, reward + gamma * next_value.detach())

            rl_optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            rl_optimizer.step()

        # 7) Perturbation 자체에 대한 optimizer (MSELoss 등)
        optimizer.zero_grad()
        criterion(gen_attack, gen_noattack).backward()
        optimizer.step()

        # 다음 step에서 X를 업데이트
        X = X_new.detach()

    return X, X - X_nat
