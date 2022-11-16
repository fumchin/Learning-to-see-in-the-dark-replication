import math
import os
import os.path
import random
from glob import glob
from itertools import cycle
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import imageio
import config as cfg
from dataloader import FujiDataset
from models.Unet import UNet
matplotlib.use('Agg')
if __name__ == '__main__':

    

    checkpoint_path = os.path.join('/home/fumchin/data/cv/final/src/stored_data', cfg.model_name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    train_dataset = FujiDataset(cfg.root, cfg.train_preprocess_file_name, None)
    validation_dataset = FujiDataset(cfg.root, cfg.val_preprocess_file_name, None)
    test_dataset = FujiDataset(cfg.root, cfg.test_file_name, None)
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=cfg.batch_size)
    validation_dataloader = DataLoader(validation_dataset, shuffle=True, batch_size=cfg.batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=cfg.batch_size)

    model_eval = UNet().to(device)
    best_checkpoint = torch.load(os.path.join(checkpoint_path, 'best_model.pt'))
    model_eval.load_state_dict(best_checkpoint['model_state_dict'])
    model_eval.eval()
    # loss_fn = nn.L1Loss()
    demo_path = os.path.join(checkpoint_path, 'demo_val')
    if not os.path.exists(os.path.join(demo_path)):
        os.makedirs(demo_path)
    with torch.no_grad():
        
        for count, (input, target) in enumerate(validation_dataloader):
        # for count, (features, label) in enumerate(train_dataloader):

            input = input.to(device)
            target = target.to(device)
            pred = model_eval(input)
            # print(pred.shape[0])
            for index in range(pred.shape[0]):
                src_img = pred[index, :, :,:]
                src_img = src_img.permute(1, 2, 0)
                src_img = src_img.detach().cpu().numpy()
                # src_img = Image.fromarray((src_img*255).astype(np.uint8))
                # src_img.save(os.path.join(demo_path, 'src_' + str(count) + '_' + str(index) + ".jpeg"))
                imageio.imwrite(os.path.join(demo_path, 'src_' + str(count) + '_' + str(index) + ".png"),src_img) 
                
                target_img = target[index, :, :,:]
                target_img = torch.tensor(target_img)
                target_img = target_img.permute(1, 2, 0)
                target_img = target_img.detach().cpu().numpy()
                # target_img = Image.fromarray((target_img*255).astype(np.uint8))
                imageio.imwrite(os.path.join(demo_path, 'target_' + str(count) + '_' + str(index) + ".png"),target_img) 

