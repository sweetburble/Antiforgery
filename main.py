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
from logger import setup_logger
from concurrent.futures import ThreadPoolExecutor

def main():
    logger = setup_logger() # 로깅을 위한 logger 설정
    total_time = time.time() # 시작 시간 저장

    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    # parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--image_size', type=int, default=256, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')

    # Data configuration
    parser.add_argument('--batch_size', type=int, default=8, help='mini-batch size') # 배치 크기 최적화
    parser.add_argument('--attack_iters', type=int, default=100)

    parser.add_argument('--resume_iters', type=int, default=200000, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])

    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    parser.add_argument('--num_workers', type=int, default=4)
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
    logger.info(f"************ celeba_loader 실행 시간: {execution_time} 초")
    print(f"************ celeba_loader 실행 시간: {execution_time} 초")

    # 모델 설정 및 로드
    start_time = time.time()
    G = Generator(config.g_conv_dim, config.c_dim, config.g_repeat_num).cuda()

    D = Discriminator(config.image_size, config.d_conv_dim, config.c_dim, config.d_repeat_num).cuda()

    print('Loading the trained models...')
    logger.info('Loading the trained models...')
    
    # load weights
    print('Loading the trained models from step {}...'.format(config.resume_iters))
    logger.info('Loading the trained models from step {}...'.format(config.resume_iters))

    G_path = os.path.join(config.model_save_dir, '{}-G.ckpt'.format(config.resume_iters))
    D_path = os.path.join(config.model_save_dir, '{}-D.ckpt'.format(config.resume_iters))
    
    load_model_weights(G, G_path)
    D.load_state_dict(torch.load(D_path, map_location='cuda'))
    print("loading model successful")
    logger.info("loading model successful")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"*********** 모델 및 가중치 load 실행 시간: {execution_time} 초")
    logger.info(f"*********** 모델 및 가중치 load 실행 시간: {execution_time} 초")

    # 이미지 반복하여 처리 
    start_time = time.time()

    # 평균적인 이미지 한 장 처리 시간 측정을 위해
    execution_time_one_image = 0.0

    l2_error_total, ssim_total, psnr_total = 0.0, 0.0, 0.0

    # ThreadPoolExecutor를 사용한 멀티스레딩 처리
    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        futures = [
            executor.submit(process_batch, x_real_batch.cuda(), c_org_batch.cuda(), config, G)
            for x_real_batch, c_org_batch in celeba_loader
        ]

        for future in futures:
            l2_error_local, ssim_local, psnr_local = future.result()
            l2_error_total += l2_error_local
            ssim_total += ssim_local
            psnr_total += psnr_local

            print(f"Processed batch with L2 Error: {l2_error_local:.4f}, SSIM: {ssim_local:.4f}, PSNR: {psnr_local:.4f}")

    n_samples = len(celeba_loader.dataset)

    print(f'Total processed images: {n_samples}')
    
    print(f'L2 Error: {l2_error_total / n_samples:.4f}')
    
    print(f'SSIM: {ssim_total / n_samples:.4f}')
    
    print(f'PSNR: {psnr_total / n_samples:.4f}')

    # # 가로로 이어 붙인 이미지를 저장
    # x_concat = torch.cat(x_fake_list, dim=3)  # dim=3은 가로 방향으로 이어 붙임
    # result_path = os.path.join(config.result_dir, f'{i+1}-images.jpg')
    # save_image(denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)

    # image_end = time.time()
    # execution_image = image_end - image_start
    # print(f"*********** {i+1}번째 이미지 처리 시간: {execution_image} 초")   
    # logger.info(f"*********** {i+1}번째 이미지 처리 시간: {execution_image} 초")

    # end_time = time.time()
    # execution_time = end_time - start_time
    
    print(f"*********** 이미지 처리 전체 실행 시간: {execution_time} 초")
    logger.info(f"*********** 이미지 처리 전체 실행 시간: {execution_time} 초")
    
    # print(f"*********** 평균적인 이미지 한 장 처리 시간: {execution_time_one_image / 50} 초")
    # logger.info(f"*********** 평균적인 이미지 한 장 처리 시간: {execution_time_one_image / 50} 초")        

    # # Print metrics
    # print('{} images.L2 error: {}. ssim: {}. psnr: {}. n_dist: {}'.format(n_samples,
    #                                                                  l2_error / n_samples,
    #                                                                  ssim / n_samples,
    #                                                                  psnr / n_samples,
    #                                                                 float(n_dist) / n_samples))
    
    total_end_time = time.time()
    total = total_end_time-total_time
    print(f"*********** main 실행시간: {total} 초")
    logger.info(f"*********** main 실행시간: {total} 초")    

'''
여기서부터 병렬/분산 처리를 위한 코드 추가

'''


def process_batch(x_real, c_org, config, G):
    """이미지 배치를 처리하는 함수"""
    x_real = x_real.cuda(non_blocking=True)
    c_org = c_org.cuda(non_blocking=True)

    c_trg_list = create_labels(c_org, config.c_dim, 'CelebA', config.selected_attrs)

    l2_error_local, ssim_local, psnr_local = 0.0, 0.0, 0.0

    for idx in range(len(x_real)):
        x_real_single = x_real[idx:idx+1]
        c_trg = [c[idx:idx+1] for c in c_trg_list]

        # LAB 공격 수행
        x_adv, pert = lab_attack(x_real_single, c_trg, G, iter=config.attack_iters)

        for c in c_trg:
            with torch.no_grad():
                gen_noattack, _ = G(x_real_single, c)
                gen_adv, _ = G(x_adv, c)

            # 품질 평가
            l2_error_local += F.mse_loss(gen_adv, gen_noattack).item()
            ssim_score, psnr_score = compare(denorm(gen_adv), denorm(gen_noattack))
            ssim_local += ssim_score
            psnr_local += psnr_score

    return l2_error_local / len(x_real), ssim_local / len(x_real), psnr_local / len(x_real)

# 사용 예시
if __name__ == "__main__":
    main()