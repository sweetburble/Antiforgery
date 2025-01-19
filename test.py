# Perturbation 초기화
pert_a = torch.zeros(X_nat.shape[0], 2, X_nat.shape[2], X_nat.shape[3]).cuda().requires_grad_()
optimizer = torch.optim.Adam([pert_a], lr=1e-4, betas=(0.9, 0.999))

epsilon = 0.05
perturb_change = epsilon if action == 1 else -epsilon

# pert_a 업데이트
???

# Perturbation 적용 및 클램핑
X_lab[:, 1:, :, :] = torch.clamp(X_lab[:, 1:, :, :] + pert_a, min=-128, max=128)