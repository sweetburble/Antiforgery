import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torchvision import transforms as T
from torch.distributions import Categorical
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize
from color_space import *  # RGB-LAB 변환 모듈
from cnn import CNNEncoder, ActorCriticCNN


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

"""
TV(Total Variation) Loss (선택 사항)
아래 코드에서 tv_loss = total_variation_loss(pert_ab) 부분은 “섭동이 공간적으로 너무 급격하게 변하지 않도록” 하는 정규화의 일종
"""
def total_variation_loss(x):
    """x: [B, 2, H, W], a/b 채널에 대한 TV loss"""
    # 가로/세로 차분 절댓값 합
    loss = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])) + \
           torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    return loss


def lab_attack(X_nat, c_trg, model, epsilon=0.05, iter=100, gamma=0.99, 
               clip_range=0.2, epochs=4, patch_size=16):
    """
    - 예시: 'a채널'과 'b채널' 각각에 다양한 perturb 단계를 줄 수 있게 확장
    - TV loss 등을 추가 적용하여 갑작스러운 색 변화 방지
    """

    #########################################
    # 1) [변경점] 다양한 perturb 단계 정의
    #   예: -0.02, -0.01, 0.0, +0.01, +0.02
    #########################################
    possible_perturbs = [-0.02, -0.01, 0.00, 0.01, 0.02]

    # (a채널/b채널) 2개 × 'possible_perturbs' 개수 = (예) 5 × 2 = 10
    # 패치 개수를 합치면 총 액션 차원 = 10 × num_patches
    # 즉, patch_idx, channel_idx, step_idx를 한 번에 인코딩
    B, C, H, W = X_nat.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w

    # 최종 액션 차원 = (a/b채널×possible_perturbs 수) × 패치 개수
    action_dim = 2 * len(possible_perturbs) * num_patches

    # ---------------------------------------
    # 1) ActorCriticCNN 초기화
    rl_model = ActorCriticCNN(action_dim=action_dim, encoder_out_dim=256).cuda()
    rl_optimizer = torch.optim.Adam(rl_model.parameters(), lr=1e-4)

    # Generator 고정
    for param in model.parameters():
        param.requires_grad = False

    # LAB 변환 및 파라미터화
    X = denorm(X_nat.clone()).clamp(0, 1)
    X_lab_full = rgb2lab(X)
    X_lab_full[:, 0, :, :] = torch.clamp(X_lab_full[:, 0, :, :].clone(), min=0.0, max=100.0)
    X_lab_full[:, 1, :, :] = torch.clamp(X_lab_full[:, 1, :, :].clone(), min=-128.0, max=128.0)
    X_lab_full[:, 2, :, :] = torch.clamp(X_lab_full[:, 2, :, :].clone(), min=-128.0, max=128.0)

    # a/b 채널만 Parameter로 관리
    pert_ab = nn.Parameter(X_lab_full[:, 1:, :, :].clone())
    pert_optimizer = torch.optim.Adam([pert_ab], lr=1e-4)

    for step in range(iter):
        # 현재 LAB (L은 고정, a/b는 learnable)
        X_lab_full[:, 1:, :, :] = pert_ab
        
        # -----------------------------
        # 부분 1) CNN 인코더로 state를 구한다
        # -----------------------------
        # CNN에 넣으려면 shape=[B, 3, H, W] 여야 함
        # (기존 a/b만 Param으로 관리했지만, L 채널도 함께 CNN에 넣어야 "전체 이미지" 상태 파악 가능)
        # 따라서 X_lab_full을 그대로 rl_model에 입력
        state_img = X_lab_full.detach()
        # shape이 [B, 3, H, W], 3=(L,a,b)라고 가정
        # 혹시 실제 shape가 [B, 1 or 2, ...] 라면, L채널도 합쳐서 3채널 맞춰야 함


        # Actor-Critic으로 액션 선택
        policy_logits, value = rl_model(state_img)
        dist = Categorical(logits=policy_logits)
        action = dist.sample() # [B] 형태이므로 B=1인 경우만

        old_log_prob = dist.log_prob(action).detach()
        old_value = value.detach()

        #######################################################
        # 2) action을 파싱하여 
        #   (어느 채널(a/b), 어느 패치, 어느 단계) 인지 구분
        #######################################################
        # 총 action_dim = 2(채널 수) × len(possible_perturbs) × num_patches
        # 세로로 묶으면 
        #   channel_idx = (action // (len(possible_perturbs) * num_patches)) % 2  # 0 => a채널, 1 => b채널
        #   step_idx    = (action // num_patches) % len(possible_perturbs)       # 몇 번째 perturb 값인지
        #   patch_idx   = action % num_patches

        channel_idx = (action.item() // (len(possible_perturbs) * num_patches)) % 2
        step_idx    = (action.item() // num_patches) % len(possible_perturbs)
        patch_idx   = action.item() % num_patches

        perturb_val = possible_perturbs[step_idx]

        py = patch_idx // num_patches_w
        px = patch_idx %  num_patches_w
        y0, y1 = py * patch_size, (py + 1) * patch_size
        x0, x1 = px * patch_size, (px + 1) * patch_size

        # 선택된 channel(a or b)의 patch 영역에 perturb 적용
        with torch.no_grad():
            temp_pert = pert_ab.data.clone()
            temp_pert[:, channel_idx, y0:y1, x0:x1] += perturb_val
            temp_pert = temp_pert.clamp(-128, 128)
            pert_ab.data.copy_(temp_pert)
        
        # Perturbation Range 계산
        pert_range_min = pert_ab.min().item()
        pert_range_max = pert_ab.max().item()

        # LAB → RGB 변환
        X_lab_full[:, 1:, :, :] = pert_ab
        X_new_rgb = lab2rgb(X_lab_full)
        X_new_rgb = X_new_rgb.clamp(0, 1)  # 혹시 모를 범위 초과 방지
        X_new = T.Normalize([0.5]*3, [0.5]*3)(X_new_rgb)

        # Generator 추론
        gen_noattack, _ = model(X_nat, c_trg[step % len(c_trg)])
        gen_attack, _   = model(X_new, c_trg[step % len(c_trg)])

        # PSNR/SSIM 계산
        ssim_input, psnr_input = compare(X_nat, X_new)
        ssim_attack, psnr_attack = compare(gen_noattack, gen_attack)
        reward = -(psnr_attack + ssim_attack) + (psnr_input + ssim_input)

        # PPO 업데이트: advantage 등 계산
        _, next_value = rl_model(state_img)
        advantage = reward + gamma * next_value.detach() - old_value

        for epoch in range(epochs):
            new_policy_logits, new_value = rl_model(state_img)
            dist_new = Categorical(logits=new_policy_logits)
            new_log_prob = dist_new.log_prob(action)
            
            ratio = torch.exp(new_log_prob - old_log_prob)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-clip_range, 1+clip_range) * advantage
            
            policy_loss = -torch.min(surr1, surr2)
            value_loss  = F.mse_loss(new_value, advantage + old_value)
            
            rl_loss = policy_loss + value_loss
            rl_optimizer.zero_grad()
            rl_loss.backward()
            rl_optimizer.step()

        # 3) [선택 사항] TV loss 등으로 섭동 부드럽게 만들기
        #    (너무 patch 단위로 색이 튀지 않게 추가 손실을 줄 수 있음)
        # tv_loss = total_variation_loss(pert_ab)
        # pert_optimizer.zero_grad()
        # tv_loss.backward()
        # pert_optimizer.step()

        # Loss, Perturbation Range, epsilon Range 출력
        print(f"[Iteration {step}] Reward : {reward:.4f}, "
              f"PSNR Input: {psnr_input:.4f}, SSIM Input: {ssim_input:.4f}, "
              f"PSNR Attack: {psnr_attack:.4f}, SSIM Attack: {ssim_attack:.4f}, "
              f"Perturbation Range: {pert_range_min:.4f} to {pert_range_max:.4f}, "
              )

    # 최종 결과물
    return X_new.detach(), (X_new.detach() - X_nat)
