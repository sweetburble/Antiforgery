import torch

def rgb2xyz(rgb):
    mask = (rgb > .04045).type(torch.FloatTensor)
    if rgb.is_cuda:
        mask = mask.cuda()
    rgb = (((rgb + .055) / 1.055) ** 2.4) * mask + rgb / 12.92 * (1 - mask)
    x = .412453 * rgb[:, 0, :, :] + .357580 * rgb[:, 1, :, :] + .180423 * rgb[:, 2, :, :]
    y = .212671 * rgb[:, 0, :, :] + .715160 * rgb[:, 1, :, :] + .072169 * rgb[:, 2, :, :]
    z = .019334 * rgb[:, 0, :, :] + .119193 * rgb[:, 1, :, :] + .950227 * rgb[:, 2, :, :]
    out = torch.cat((x[:, None, :, :], y[:, None, :, :], z[:, None, :, :]), dim=1)
    out = torch.clamp(out, min=0, max=1)
    return out


def xyz2rgb(xyz):
    r = 3.24048134 * xyz[:, 0, :, :] - 1.53715152 * xyz[:, 1, :, :] - 0.49853633 * xyz[:, 2, :, :]
    g = -0.96925495 * xyz[:, 0, :, :] + 1.87599 * xyz[:, 1, :, :] + .04155593 * xyz[:, 2, :, :]
    b = .05564664 * xyz[:, 0, :, :] - .20404134 * xyz[:, 1, :, :] + 1.05731107 * xyz[:, 2, :, :]
    rgb = torch.cat((r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]), dim=1)
    rgb = torch.clamp(rgb, min=0, max=1)
    return rgb

# 코드 개선 버전?
# def xyz2rgb(xyz):
#     # 1) XYZ → (선형) RGB 변환
#     r = 3.24048134 * xyz[:, 0, :, :] - 1.53715152 * xyz[:, 1, :, :] - 0.49853633 * xyz[:, 2, :, :]
#     g = -0.96925495 * xyz[:, 0, :, :] + 1.87599 * xyz[:, 1, :, :] + .04155593 * xyz[:, 2, :, :]
#     b = .05564664 * xyz[:, 0, :, :] - .20404134 * xyz[:, 1, :, :] + 1.05731107 * xyz[:, 2, :, :]

#     linear_rgb = torch.cat((r[:, None, :, :], 
#                             g[:, None, :, :], 
#                             b[:, None, :, :]), dim=1)
    
#     # 2) 선형 RGB를 sRGB 감마로 보정
#     #    sRGB 표준에 따라 0.0031308 기준으로 나누어 처리
#     threshold = 0.0031308
#     under_mask  = (linear_rgb <= threshold).type(linear_rgb.dtype)
#     over_mask   = (linear_rgb > threshold).type(linear_rgb.dtype)
    
#     # under_mask인 부분은 12.92 * (linear RGB)
#     # over_mask인 부분은 1.055 * (linear RGB^(1/2.4)) - 0.055
#     srgb = under_mask * (12.92 * linear_rgb) \
#           + over_mask  * (1.055 * (linear_rgb.clamp(min=0)) ** (1.0/2.4) - 0.055)

#     srgb = torch.clamp(srgb, min=0.0, max=1.0)
#     return srgb


def xyz2lab(xyz):
    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    if xyz.is_cuda:
        sc = sc.cuda()
    xyz_scale = xyz / sc
    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if xyz_scale.is_cuda:
        mask = mask.cuda()
    xyz_int = xyz_scale ** (1 / 3.) * mask + (7.787 * xyz_scale + 16. / 116.) * (1 - mask)
    L = 116. * xyz_int[:, 1, :, :] - 16.
    a = 500. * (xyz_int[:, 0, :, :] - xyz_int[:, 1, :, :])
    b = 200. * (xyz_int[:, 1, :, :] - xyz_int[:, 2, :, :])
    out = torch.cat((L[:, None, :, :], a[:, None, :, :], b[:, None, :, :]), dim=1)
    return out

def lab2xyz(lab):
    y_int = (lab[:, 0, :, :] + 16.) / 116.
    x_int = (lab[:, 1, :, :] / 500.) + y_int
    z_int = y_int - (lab[:, 2, :, :] / 200.)
    x_int = torch.clamp(x_int, min=0)
    y_int = torch.clamp(y_int, min=0)
    z_int = torch.clamp(z_int, min=0)
    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None].to(lab.device)
    out = torch.cat((x_int[:, None, :, :], y_int[:, None, :, :], z_int[:, None, :, :]), dim=1) * sc
    return out

def rgb2lab(rgb):
    return xyz2lab(rgb2xyz(rgb))

def lab2rgb(lab):
    return xyz2rgb(lab2xyz(lab))
