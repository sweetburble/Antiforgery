from math import e
import torch
import argparse
import os
from torchvision.utils import save_image
import torch
import torch.nn.functional as F

from color_space import *
from data_loader import get_loader, CelebA
from utils import *
from gpu_logger import ResourceMonitor

from model import Generator, Discriminator

import time
import threading
from time_logger import setup_logger

'''
분산 처리를 위해 추가한 import
'''
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    # CPU/GPU 모니터링 시작
    resource_monitor = ResourceMonitor(sampling_interval=1)
    monitoring_thread = threading.Thread(target=resource_monitor.collect_metrics, daemon=True)
    monitoring_thread.start()

    time_logger = setup_logger() # 프로젝트 소요시간 로깅을 위한 logger 설정
    start_time = time.time() # 프로젝트 시작 시간
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

    # Data configuration
    parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size')
    parser.add_argument('--attack_iters', type=int, default=100)

    parser.add_argument('--resume_iters', type=int, default=200000, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])

    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])

    parser.add_argument('--celeba_image_dir', type=str, default='../test_rl/data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='../test_rl/data/celeba/list_attr_celeba.txt')
    parser.add_argument('--model_save_dir', type=str, default='../test_rl/stargan_celeba_256/models')
    parser.add_argument('--result_dir', type=str, default='results')

    args = parser.parse_args()

    # GPU 개수 확인
    world_size = torch.cuda.device_count()
    print(f'Found {world_size} GPUs!')
    
    try:
        # 분산 학습 시작
        mp.spawn(
            train_model,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    finally:
        # 모니터링 종료 및 통계 기록
        resource_monitor.do_run = False
        monitoring_thread.join()
        resource_monitor.log_statistics()

    total_time = time.time() - start_time
    print(f"전체 프로젝트 소요 시간 : {total_time} 초")
    time_logger.info(f"전체 프로젝트 소요 시간 : {total_time} 초")



'''
여기서부터 병렬/분산 처리를 위한 코드 추가
'''
def setup(rank, world_size):
    """분산 학습 환경 설정"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """분산 학습 환경 정리"""
    dist.destroy_process_group()

from torchvision import transforms as T

def train_model(rank, world_size, args):
    time_logger = setup_logger() if (rank == 0) else None # 프로젝트 소요시간 로깅을 위한 logger 설정
    
    # 각 GPU에서 실행될 학습 함수
    print(f"Running training on rank {rank}.")

    setup(rank, world_size)

    # 모델 설정 시작
    model_load_start = time.time()
    
    # rank 0에서만 실행시간을 로깅한다
    if rank == 0:
        time_logger.info(f"프로세스 시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # GPU 설정
    torch.cuda.set_device(rank)
    
    # 모델 초기화
    G = Generator(args.g_conv_dim, args.c_dim, args.g_repeat_num)
    D = Discriminator(args.image_size, args.d_conv_dim, args.c_dim, args.d_repeat_num)
    
    G = G.cuda(rank)
    D = D.cuda(rank)
    
    # DDP 래핑
    G = DDP(G, device_ids=[rank])
    D = DDP(D, device_ids=[rank])
    
    # 가중치 로드
    G_path = os.path.join(args.model_save_dir, f'{args.resume_iters}-G.ckpt')
    D_path = os.path.join(args.model_save_dir, f'{args.resume_iters}-D.ckpt')
    
    load_model_weights(G.module, G_path)
    D.module.load_state_dict(torch.load(D_path, map_location=f'cuda:{rank}', weights_only=True))
    
    if rank == 0:
        print(f"모델 로드 완료 시간: {time.time() - model_load_start:.2f}초")
        time_logger.info(f"모델 로드 완료 시간: {time.time() - model_load_start:.2f}초")

    
    data_setting_start = time.time()

    # data_loader.py의 get_loader 흉내
    """Build and return a data loader."""
    transform = []
    transform.append(T.CenterCrop(args.celeba_crop_size))
    transform.append(T.Resize((args.image_size, args.image_size)))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    # 데이터셋 전체 크기 확인
    celeba_dataset = CelebA(args.celeba_image_dir, args.attr_path, args.selected_attrs,
                          transform, args.mode)
    
    # 각 GPU별 처리할 이미지 인덱스 계산
    total_images = 50  # 전체 처리할 이미지 수
    images_per_gpu = total_images // world_size  # GPU당 처리할 이미지 수
    start_idx = rank * images_per_gpu
    end_idx = start_idx + images_per_gpu
    
    # GPU별 데이터셋 분할
    indices = list(range(start_idx, end_idx))
    subset = torch.utils.data.Subset(celeba_dataset, indices)
    
    # 데이터로더 생성
    celeba_loader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8, # worker 수 증가
        pin_memory=True,
        persistent_workers=True # worker 재사용
    )

    l2_error, ssim, psnr = 0.0, 0.0, 0.0
    n_samples, n_dist = 0, 0

    if (rank == 0): 
        print(f"Dataset, DataLoader 설정 시간 : {time.time() - data_setting_start:.2f}초")
        time_logger.info(f"Dataset, DataLoader 설정 시간 : {time.time() - data_setting_start:.2f}초")

    total_image_process_start = time.time() # 이미지 한 장당 소요되는 시간 측정을 위해

    # 이미지 처리
    for i, (x_real, c_org) in enumerate(celeba_loader):
        if i >= 25:  # 각 GPU당 25개 이미지만 처리
            break
            
        batch_start = time.time()
        
        # Prepare input images and target domain labels.
        x_real = x_real.cuda(rank)
        c_trg_list = create_labels(c_org, args.c_dim, 'CelebA', args.selected_attrs)

        x_fake_list = [x_real]

        # generate adv in lab space
        x_adv, pert = lab_attack(x_real, c_trg_list, G, iter=args.attack_iters)

        x_fake_list.append(x_adv)

        if (rank == 0):
            print(f"처음 이미지를 처리하는 데 걸리는 시간 : {time.time() - batch_start}")
            time_logger.info(f"처음 이미지를 처리하는 데 걸리는 시간 : {time.time() - batch_start}")

        # 결과 저장 및 평가
        for idx, c_trg in enumerate(c_trg_list):
            print(f'GPU {rank}: Processing image {start_idx + i}, class {idx}')
            
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

        # GPU 별 결과 저장
        x_concat = torch.cat(x_fake_list, dim=3)
        result_path = os.path.join(args.result_dir, f'GPU_{rank}_image{start_idx + i + 1}.jpg')
        save_image(denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)

        if rank == 0:
            image_end = time.time()
            execution_image = image_end - batch_start
            print(f"*********** {i+1}번째 이미지 처리 시간 : {execution_image} 초")
            time_logger.info(f"{i+1}번째 이미지 처리 시간 : {execution_image}초")
    
    # 메트릭 동기화
    dist.all_reduce(torch.tensor([l2_error]).cuda(rank), op=dist.ReduceOp.SUM)
    dist.all_reduce(torch.tensor([ssim]).cuda(rank), op=dist.ReduceOp.SUM)
    dist.all_reduce(torch.tensor([psnr]).cuda(rank), op=dist.ReduceOp.SUM)
    dist.all_reduce(torch.tensor([n_samples]).cuda(rank), op=dist.ReduceOp.SUM)
    dist.all_reduce(torch.tensor([n_dist]).cuda(rank), op=dist.ReduceOp.SUM)

    if rank == 0:
        end_time = time.time()
        total_time = (end_time - total_image_process_start)
        avg_time = (end_time - total_image_process_start) / 25

        print(f"평균적인 이미지 한 장 처리 시간 : {avg_time}초")
        time_logger.info(f"평균적인 이미지 한 장 처리 시간 : {avg_time}초")

        print(f"전체 이미지 처리 시간 : {total_time}초")
        time_logger.info(f"전체 이미지 처리 시간 : {total_time}초")

        print(f'{n_samples} images.\nL2 error : {l2_error/n_samples}. '
              f'ssim : {ssim/n_samples}. psnr : {psnr/n_samples}. '
              f'n_dist : {float(n_dist)/n_samples}')
        time_logger.info(f'{n_samples} images.\nL2 error : {l2_error/n_samples}. '
              f'ssim : {ssim/n_samples}. psnr : {psnr/n_samples}. '
              f'n_dist : {float(n_dist)/n_samples}')
    
    cleanup()

# 사용 예시
if __name__ == "__main__":
    main()