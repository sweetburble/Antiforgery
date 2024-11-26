from calendar import c
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
import ray
from logger import setup_logger

def main():
    logger = setup_logger() # 로깅을 위한 logger 설정
    total_time = time.time() # 시작 시간 저장

    ray.init() # Ray 초기화

    # 설정값 초기화
    config = {
        'celeba_image_dir': 'C:/Users/Bandi/Desktop/Fork/AntiForgery/data/celeba/images',
        'attr_path': 'C:/Users/Bandi/Desktop/Fork/AntiForgery/data/celeba/list_attr_celeba.txt',
        'selected_attrs': ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'],
        'batch_size': 4,
        'image_size': 256,
        'g_conv_dim': 64,
        'c_dim': 5,
        'g_repeat_num': 6,
        'num_workers': 4,
        'model_path': 'stargan_celeba_256/models/200000-D.ckpt',  # 생성자 가중치 경로 추가
    }

    # 데이터 로더 초기화
    # 데이터 로더 설정
    celeba_loader = get_loader(
        image_dir=config['celeba_image_dir'],
        attr_path=config['attr_path'],
        selected_attrs=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"],
        mode="test"
    )

    # data loader
    start_time = time.time()
    
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"************ celeba_loader 실행 시간: {execution_time} 초")
    print(f"************ celeba_loader 실행 시간: {execution_time} 초")

    # 모델 설정 및 로드
    start_time = time.time()

    # 모델 설정
    G = Generator(conv_dim=64, c_dim=5, repeat_num=6).cuda()
    D = Discriminator(image_size=256, conv_dim=64, c_dim=5, repeat_num=6).cuda()
    G.load_state_dict(torch.load("path/to/G.ckpt"))
    D.load_state_dict(torch.load("path/to/D.ckpt"))

    # 이미지 처리 루프
    for i, (x_real, c_org) in enumerate(celeba_loader):
        x_real = x_real.cuda()
        c_trg_list = create_labels(c_org, c_dim=5, dataset='CelebA', selected_attrs=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"])

        # Ray 태스크로 LAB 공격 수행
        futures = [lab_attack_ray.remote(x_real[i:i+1], c_trg_list[i], G) for i in range(len(x_real))]
        results = ray.get(futures)  # 병렬 처리 결과 가져오기

        x_adv_list = [res[0] for res in results]
        pert_list = [res[1] for res in results]

        # 결과 저장
        for j, x_adv in enumerate(x_adv_list):
            result_path = os.path.join("results", f"{i}_{j}_adv.jpg")
            save_image(denorm(x_adv.cpu()), result_path)
    
    total_end_time = time.time()
    total = total_end_time-total_time
    print(f"*********** main 실행시간: {total} 초")
    logger.info(f"*********** main 실행시간: {total} 초")    

'''
여기서부터 병렬/분산 처리를 위한 코드 추가

'''

@ray.remote
def process_image(x_real, c_trg_list, config):
    """
    Ray 작업: 이미지 처리 (LAB 공격 수행)
    """
    # GPU 사용 여부 확인 및 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 각 worker 내에서 모델 초기화
    G = Generator(config['g_conv_dim'], config['c_dim'], config['g_repeat_num']).to(device)
    G.eval()

    # 결과 저장 리스트
    x_adv_list = []
    
    # 각 도메인 레이블에 대해 LAB 공격 수행
    for c_trg in c_trg_list:
        c_trg = c_trg.to(device)
        x_real = x_real.to(device)
        x_adv, pert = lab_attack(x_real, c_trg, G)
        x_adv_list.append((x_adv.cpu(), pert.cpu()))  # 결과를 CPU로 이동하여 반환

    return x_adv_list

# 사용 예시
if __name__ == "__main__":
    main()