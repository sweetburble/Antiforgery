import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms as T
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize
from color_space import *  # RGB-LAB 변환 모듈

def load_model_weights(model, path):
    """모델의 가중치를 불러오는 함수"""
    pretrained_dict = torch.load(path, map_location='cuda', weights_only=True)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'preprocessing' not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)

def denorm(x):
    """[-1, 1] 범위를 [0, 1]로 변환"""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def label2onehot(labels, dim):
    """레이블 인덱스를 원-핫 벡터로 변환"""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

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

def compare(img1, img2):
    """이미지 비교 (SSIM 및 PSNR 계산)"""
    img1_np = img1.squeeze(0).cpu().numpy()
    img2_np = img2.squeeze(0).cpu().numpy()
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

def lab_attack(X_nat, c_trg, model, epsilon=0.05, iter=100):
    """LAB 공간에서의 공격"""
    criterion = nn.MSELoss().cuda()
    pert_a = torch.zeros(X_nat.shape[0], 2, X_nat.shape[2], X_nat.shape[3]).cuda().requires_grad_()
    optimizer = torch.optim.Adam([pert_a], lr=1e-4, betas=(0.9, 0.999))

    X = denorm(X_nat.clone())

    for i in range(iter):
        X_lab = rgb2lab(X).cuda()
        pert = torch.clamp(pert_a, min=-epsilon, max=epsilon)
        X_lab[:, 1:, :, :] = X_lab[:, 1:, :, :] + pert
        X_lab = torch.clamp(X_lab, min=-128, max=128)

        try:
            X_new = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(lab2rgb(X_lab))
        except Exception as e:
            print(f"Error in LAB to RGB conversion: {e}")
            break

        with torch.no_grad():
            gen_noattack, _ = model(X_nat, c_trg[i % len(c_trg)])

        gen_stargan, _ = model(X_new, c_trg[i % len(c_trg)])
        loss = -criterion(gen_stargan, gen_noattack)

        if torch.isnan(loss):
            print(f"NaN detected in Loss at iteration {i}. Stopping attack.")
            break

        optimizer.zero_grad()
        loss.backward()

        if torch.isnan(pert_a.grad).any():
            print(f"NaN detected in gradient at iteration {i}. Resetting gradient.")
            pert_a.grad = torch.zeros_like(pert_a.grad)
            continue

        optimizer.step()

        if torch.isnan(pert_a).any():
            print(f"NaN detected in perturbation after optimizer step at iteration {i}. Stopping.")
            break

        # print(f"[lab_attack] Iter {i}: Loss={loss.item()}, Perturbation Range: {pert.min().item()} to {pert.max().item()}")
    return X_new, X_new - X
