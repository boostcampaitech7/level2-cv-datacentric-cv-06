import os
import os.path as osp
import time
import math
import numpy as np
from datetime import timedelta
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

import torch
from torch import cuda
from torch.utils.data import DataLoader, Subset
from torch.optim import lr_scheduler
from tqdm import tqdm
import wandb
import random

from east_dataset import EASTDataset
from dataset_filtered import SceneTextDataset
from model import EAST

def set_seed(seed=42):
    """Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=5)
    
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval):
    
    print("Initializing wandb...")
    wandb.init(
        project="OCR",
        config={
            "learning_rate": learning_rate,
            "architecture": "EAST",
            "dataset": "Scene Text",
            "epochs": max_epoch,
            "batch_size": batch_size,
            "image_size": image_size,
            "input_size": input_size,
        }
    )

    print("Loading dataset...")
    base_dataset = SceneTextDataset(
        data_dir,
        split='filtered/train_filtered_both',
        image_size=image_size,
        crop_size=input_size,
    )
    print(f"Dataset size: {len(base_dataset)}")

    print("Splitting indices...")
    indices = list(range(len(base_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(np.floor(0.9 * len(base_dataset)))
    train_indices, val_indices = indices[:split], indices[split:]
    
    print(f"Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}")

    train_base = Subset(base_dataset, train_indices)
    val_base = Subset(base_dataset, val_indices)
    print(f"Base subset sizes - Train: {len(train_base)}, Val: {len(val_base)}")

    print("Converting to EAST datasets...")
    train_dataset = EASTDataset(train_base)
    val_dataset = EASTDataset(val_base)
    print(f"EAST dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    print("Creating data loaders...")
    num_batches = math.ceil(len(train_dataset) / batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    val_num_batches = math.ceil(len(val_dataset) / batch_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    print("Initializing model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()

    print("Loading pretrained model...")
    checkpoint = torch.load(osp.join(model_dir,"Textgen_to_opendata_epoch_30.pth"))
    model.load_state_dict(checkpoint)
    model.to(device)
    
    print("Setting up optimizer and scheduler...")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(max_epoch):
        torch.cuda.empty_cache()
        
        # Training phase
        model.train()
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for batch_idx, (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(train_loader):
                if batch_idx % 5 == 0:
                    pbar.set_description('[Epoch {}]'.format(epoch + 1))
                    pbar.update(5)

                img = img.to(device, non_blocking=True)
                gt_score_map = gt_score_map.to(device, non_blocking=True)
                gt_geo_map = gt_geo_map.to(device, non_blocking=True)
                roi_mask = roi_mask.to(device, non_blocking=True)

                optimizer.zero_grad()
                
                # Mixed precision 제거하고 일반 training
                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                if batch_idx % 5 == 0:  # 5번째 배치마다 로깅
                    wandb.log({
                        'batch_loss': loss_val,
                        'cls_loss': extra_info['cls_loss'],
                        'angle_loss': extra_info['angle_loss'],
                        'iou_loss': extra_info['iou_loss'],
                        'learning_rate': optimizer.param_groups[0]['lr']
                    })

                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 
                    'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)
        
        # Validation Phase
        model.eval()
        val_epoch_loss = 0
        val_metrics = {'cls_loss': 0, 'angle_loss': 0, 'iou_loss': 0}
        
        with torch.no_grad():
            for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                img = img.to(device, non_blocking=True)
                gt_score_map = gt_score_map.to(device, non_blocking=True)
                gt_geo_map = gt_geo_map.to(device, non_blocking=True)
                roi_mask = roi_mask.to(device, non_blocking=True)
                
                # forward pass만 수행
                pred_score_map, pred_geo_map = model(img)
                loss, extra_info = model.criterion(gt_score_map, pred_score_map, gt_geo_map, pred_geo_map, roi_mask)
                
                val_epoch_loss += loss.item()
                val_metrics['cls_loss'] += extra_info['cls_loss']
                val_metrics['angle_loss'] += extra_info['angle_loss']
                val_metrics['iou_loss'] += extra_info['iou_loss']

        val_epoch_loss /= val_num_batches
        val_metrics = {k: v / val_num_batches for k, v in val_metrics.items()}

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model_state = model.state_dict().copy()
            
            if not osp.exists(model_dir):
                os.makedirs(model_dir)
            best_model_path = osp.join(model_dir, 'best_model.pth')
            torch.save(best_model_state, best_model_path)
            wandb.save(best_model_path)
            print(f'New best model saved! Validation loss: {val_epoch_loss:.4f}')

        wandb.log({
            'val_loss': val_epoch_loss,
            'val_cls_loss': val_metrics['cls_loss'],
            'val_angle_loss': val_metrics['angle_loss'],
            'val_iou_loss': val_metrics['iou_loss'],
            'best_val_loss': best_val_loss,
            'epoch': epoch + 1,
            'epoch_loss': epoch_loss / num_batches,
            'epoch_time': time.time() - epoch_start
        })

        print(f'Validation loss: {val_epoch_loss:.4f} | Best validation loss: {best_val_loss:.4f}')
        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        scheduler.step()

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)
            ckpt_fpath = osp.join(model_dir, f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            latest_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), latest_fpath)
            wandb.save(ckpt_fpath)
            wandb.save(latest_fpath)

    wandb.finish()

def main(args):
    set_seed(42)
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)