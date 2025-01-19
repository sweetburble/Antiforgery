from math import e
import torch
import numpy as np
import argparse
import os
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F

from color_space import *
from data_loader import get_loader
from utils import *

from model import Generator, Discriminator

import time
import psutil


def main():
    total_time = time.time() # 시작 시간 저장
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--image_size', type=int, default=256, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')

    # Data configuration.
    parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size')
    parser.add_argument('--attack_iters', type=int, default=100)

    parser.add_argument('--resume_iters', type=int, default=200000, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])

    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])

    parser.add_argument('--celeba_image_dir', type=str, default='C:/Users/Bandi/Desktop/Fork/AntiForgery/data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='C:/Users/Bandi/Desktop/Fork/AntiForgery/data/celeba/list_attr_celeba.txt')
    parser.add_argument('--model_save_dir', type=str, default='stargan_celeba_256/models')
    parser.add_argument('--result_dir', type=str, default='results')

    config = parser.parse_args()

    os.makedirs(config.result_dir, exist_ok=True)

    # data loader
    start_time = time.time()

    celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                               config.celeba_crop_size, config.image_size, config.batch_size,
                               'CelebA', config.mode, config.num_workers)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"************ celeba_lader 실행 시간: {execution_time} 초")

    # 모델 설정 및 로드
    start_time = time.time()

    G = Generator(config.g_conv_dim, config.c_dim, config.g_repeat_num)
    D = Discriminator(config.image_size, config.d_conv_dim, config.c_dim, config.d_repeat_num)
    G.cuda()
    D.cuda()
    
    # load weights
    print('Loading the trained models from step {}...'.format(config.resume_iters))
    G_path = os.path.join(config.model_save_dir, '{}-G.ckpt'.format(config.resume_iters))
    D_path = os.path.join(config.model_save_dir, '{}-D.ckpt'.format(config.resume_iters))
    # self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    load_model_weights(G, G_path)
    D.load_state_dict(torch.load(D_path, map_location='cuda'))
    print("loading model successful")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"*********** 모델 및 가중치 load 실행 시간: {execution_time} 초")

    l2_error, ssim, psnr = 0.0, 0.0, 0.0
    n_samples, n_dist = 0, 0

    # 이미지 반복하여 처리 
    start_time = time.time()

    # 평균적인 이미지 한 장 처리 시간 측정을 위해
    execution_time_one_image = 0.0

    for i, (x_real, c_org) in enumerate(celeba_loader):
        # 10개 이미지만 처리
        if i == 10: 
            break

        image_start = time.time()

        # Prepare input images and target domain labels.
        x_real = x_real.cuda()
        c_trg_list = create_labels(c_org, config.c_dim, 'CelebA', config.selected_attrs)

        x_fake_list = [x_real]

        # generate adv in lab space
        x_adv, pert = lab_attack(x_real, c_trg_list, G, iter=config.attack_iters)

        x_fake_list.append(x_adv)

        for idx, c_trg in enumerate(c_trg_list):
            print('image', i+1, 'class', idx)
            with torch.no_grad():
                x_real_mod = x_real
                gen_noattack, gen_noattack_feats = G(x_real_mod, c_trg)

            # Metrics
            with torch.no_grad():
                gen, _ = G(x_adv, c_trg)
                # Add to lists
                x_fake_list.append(gen_noattack)
                # x_fake_list.append(perturb)
                x_fake_list.append(gen)

                l2_error += F.mse_loss(gen, gen_noattack)

                ssim_local, psnr_local = compare(denorm(gen), denorm(gen_noattack))
                ssim += ssim_local
                psnr += psnr_local

                if F.mse_loss(gen, gen_noattack) > 0.05:
                    n_dist += 1
                n_samples += 1

        # Save the translated images.
        x_concat = torch.cat(x_fake_list, dim=3)
        result_path = os.path.join(config.result_dir, '{}-images.jpg'.format(i + 1))
        save_image(denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)

        image_end = time.time()
        execution_image = image_end - image_start
        print(f"*********** {i+1}번째 이미지 처리 시간: {execution_image} 초")   
        
        execution_time_one_image += execution_image

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"*********** 이미지 처리 전체 실행 시간: {execution_time} 초")

    print(f"*********** 평균적인 이미지 한 장 처리 시간: {execution_time_one_image / 10} 초")        

    # Print metrics
    print('{} images.L2 error: {}. ssim: {}. psnr: {}. n_dist: {}'.format(n_samples,
                                                                    l2_error / n_samples,
                                                                    ssim / n_samples,
                                                                    psnr / n_samples,
                                                                    float(n_dist) / n_samples))
    
    total_end_time = time.time()
    total = total_end_time - total_time
    print(f"*********** main 실행시간: {total} 초")    

# 사용 예시
if __name__ == "__main__":
    main()