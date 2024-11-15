import os
import os.path as osp
import time
import math
import numpy as np
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
import wandb
import random

from east_dataset import EASTDataset
from dataset import SceneTextDataset
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
    parser.add_argument('--wandb_name', type=str, default="")
    
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, wandb_name):
    
    # Initialize wandb
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
    if wandb_name != "":
        wandb.run.name=wandb_name
        wandb.run.save()
    
    dataset = SceneTextDataset(
        data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # pin_memory=True,
        # persistent_workers=True,
        # prefetch_factor=4,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()

    #Load PreTrained Model 
    # checkpoint = torch.load(osp.join(model_dir,"Textgen_e30_without_clip_grad.pth"))
    # model.load_state_dict(checkpoint)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                # Log batch metrics to wandb
                wandb.log({
                    'batch_loss': loss_val,
                    'cls_loss': extra_info['cls_loss'],
                    'angle_loss': extra_info['angle_loss'],
                    'iou_loss': extra_info['iou_loss'],
                    'learning_rate': optimizer.param_groups[0]['lr']
                })

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)

        scheduler.step()

        # Log epoch metrics to wandb
        wandb.log({
            'epoch': epoch + 1,
            'epoch_loss': epoch_loss / num_batches,
            'epoch_time': time.time() - epoch_start
        })

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            # Save with epoch number
            ckpt_fpath = osp.join(model_dir, f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            
            # Also save as latest
            latest_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), latest_fpath)
            
            # Log both checkpoints to wandb
            wandb.save(ckpt_fpath)
            wandb.save(latest_fpath)

    # Close wandb run
    wandb.finish()

def main(args):
    set_seed(42)
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)