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
import demo_config as cfg
from dataloader import FujiDataset, FujiDataset_eval
from models.Unet import UNet
matplotlib.use('Agg')
if __name__ == '__main__':

    

    checkpoint_path = os.path.join('/home/fumchin/data/cv/final/src/stored_data', cfg.model_name)
    src_path = "/home/fumchin/data/cv/final/dataset/Fuji/short"
    # if not os.path.exists(checkpoint_path):
    #     os.makedirs(checkpoint_path)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    # device = "cpu"
    train_dataset = FujiDataset_eval(cfg.root, cfg.test_file_name, None)
    validation_dataset = FujiDataset_eval(cfg.root, cfg.val_file_name, None)
    test_dataset = FujiDataset_eval(cfg.root, cfg.test_file_name, None)
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1)
    validation_dataloader = DataLoader(validation_dataset, shuffle=True, batch_size=cfg.batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=cfg.batch_size)

    model_eval = UNet().to(device)
    # model_eval = UNet()
    best_checkpoint = torch.load(os.path.join(checkpoint_path, 'best_model.pt'), map_location=device)
    # best_checkpoint = torch.load(os.path.join(checkpoint_path, 'best_model.pt'))
    model_eval.load_state_dict(best_checkpoint['model_state_dict'])
    model_eval.eval()
    # loss_fn = nn.L1Loss()
    demo_path = os.path.join(checkpoint_path, 'demo_train')
    
    if not os.path.exists(os.path.join(demo_path)):
        os.makedirs(demo_path)
    with torch.no_grad():
        
        for count, (input, target, src) in enumerate(train_dataloader):
        # for count, (features, label) in enumerate(train_dataloader):

            input = input.to(device)
            target = target.to(device)
            pred = model_eval(input)
            # print(pred.shape[0])
            for index in range(pred.shape[0]):
                # src_img = input[index, :, :,:]
                # src_img = src_img.permute(1, 2, 0)
                # src_img = src_img.detach().cpu().numpy()
                # imageio.imwrite(os.path.join(demo_path, 'src_' + str(count) + '_' + str(index) + ".png"),src_img) 

                pred_img = pred[index, :, :,:]
                pred_img = pred_img.permute(1, 2, 0)
                pred_img = pred_img.detach().cpu().numpy()
                # pred_img = ((pred_img/np.max(pred_img))*255).astype(np.uint8)
                imageio.imwrite(os.path.join(demo_path, 'pred_' + str(count) + '_' + str(index) + ".png"),pred_img) 
                
                src_img = src[index, :, :,:]
                src_img = torch.tensor(src_img)
                src_img = src_img.permute(1, 2, 0)
                src_img = src_img.detach().cpu().numpy()
                # target_img = (target_img*255).astype(np.uint8)
                imageio.imwrite(os.path.join(demo_path, 'src_' + str(count) + '_' + str(index) + ".png"),src_img) 

                target_img = target[index, :, :,:]
                target_img = torch.tensor(target_img)
                target_img = target_img.permute(1, 2, 0)
                target_img = target_img.detach().cpu().numpy()
                # target_img = ((target_img/np.max(target_img))*255).astype(np.uint8)
                imageio.imwrite(os.path.join(demo_path, 'target_' + str(count) + '_' + str(index) + ".png"),target_img) 

