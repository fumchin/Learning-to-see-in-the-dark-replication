import math
import os
import os.path
import random
from glob import glob
from itertools import cycle
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchvision import transforms

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import config as cfg
from dataloader import FujiDataset
from models.Unet import UNet
matplotlib.use('Agg')


def train(model, train_dataloader, validation_dataloader, loss_fn, optimizer, device, epochs, checkpoint_path):
    val_loss_min = math.inf
    validation_psnr_max = 0
    training_loss_list = []
    validation_loss_list = []
    training_psnr_list = []
    validation_psnr_list = []
    training_ssim_list = []
    validation_ssim_list = []
    epoch_list = []
    for i in range(epochs):
        print(f"Epoch {i+1}")
        epoch_list.append(i)
        training_loss, validation_loss, training_psnr, validation_psnr, training_ssim, validation_ssim = train_single_epoch(model, train_dataloader, validation_dataloader, loss_fn, optimizer, device)
        print('training loss: ' + str(training_loss))
        print('training psnr: ' + str(training_psnr))
        print('training ssim: ' + str(training_ssim))
        print("")
        print('validation loss: ' + str(validation_loss))
        print('validation psnr: ' + str(validation_psnr))
        print('validation ssim: ' + str(validation_ssim))
        print('*'*50)
        
        training_loss_list.append(training_loss)
        validation_loss_list.append(validation_loss)
        training_psnr_list.append(training_psnr)
        validation_psnr_list.append(validation_psnr)
        training_ssim_list.append(training_ssim)
        validation_ssim_list.append(validation_ssim)
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            # 'ema_model_state_dict': ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(checkpoint_path, f'checkpoint_epoch_{i}.pt'))
        
        # if validation_loss < val_loss_min:
        if validation_psnr_max < validation_psnr:
            torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            # 'ema_model_state_dict': ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(checkpoint_path, f'best_model.pt'))

            # loss_min = validation_loss
            val_loss_min = validation_loss

            output_txt = os.path.join(checkpoint_path, 'best.txt')
            with open(output_txt, 'a') as f:
                f.write(f'training_loss: {training_loss}\n')
                f.write(f'training_psnr: {training_psnr}\n')
                f.write(f'training_ssim: {training_ssim}\n')
                f.write(f'validation_loss: {validation_loss}\n')
                f.write(f'validation_psnr: {validation_psnr}\n')
                f.write(f'validation_ssim: {validation_ssim}\n')
                f.write(f'epoch: {i}\n')
            
        plt.figure(1)
        plt.grid()
        # plt.xlim(0, 200, 1)
        plt.plot(epoch_list, training_loss_list, 'b')
        plt.plot(epoch_list, validation_loss_list, 'r')
        plt.savefig(os.path.join(checkpoint_path, f'{cfg.model_name}_loss.png'))

        plt.figure(2)
        plt.grid()
        # plt.xlim(0, 200, 1)
        plt.plot(epoch_list, training_psnr_list, 'b')
        plt.plot(epoch_list, validation_psnr_list, 'r')
        plt.savefig(os.path.join(checkpoint_path, f'{cfg.model_name}_psnr.png'))

        plt.figure(3)
        plt.grid()
        # plt.xlim(0, 200, 1)
        plt.plot(epoch_list, training_ssim_list, 'b')
        plt.plot(epoch_list, validation_ssim_list, 'r')
        plt.savefig(os.path.join(checkpoint_path, f'{cfg.model_name}_ssim.png'))

def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()
    
def train_single_epoch(model, train_dataloader, validation_dataloader, loss_fn, optimizer, device):
    training_loss = 0
    validation_loss = 0
    training_psnr = 0
    validation_psnr = 0
    training_ssim = 0
    validation_ssim = 0
    # train
    model.train()
    for input, target in tqdm(train_dataloader):
        input, target = input.to(device), target.to(device)
        prediction = model(input)
        # loss = loss_fn(prediction, target)
        loss = reduce_mean(prediction, target)
        training_loss += loss.item()
        psnr = peak_signal_noise_ratio(prediction, target)
        training_psnr += psnr.item()
        ssim = structural_similarity_index_measure(prediction, target)
        training_ssim += ssim.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    training_loss = (training_loss) / len(train_dataloader)
    training_psnr = (training_psnr) / len(train_dataloader)
    training_ssim = (training_ssim) / len(train_dataloader)
    
    model.eval()
    for input, target in tqdm(validation_dataloader):
        input, target = input.to(device), target.to(device)
        prediction = model(input)
        # loss = loss_fn(prediction, target)
        loss = reduce_mean(prediction, target)
        validation_loss += loss.item()

        psnr = peak_signal_noise_ratio(prediction, target)
        validation_psnr += psnr.item()
        ssim = structural_similarity_index_measure(prediction, target)
        validation_ssim += ssim.item()

    validation_loss = (validation_loss) / len(validation_dataloader)
    validation_psnr = (validation_psnr) / len(validation_dataloader)
    validation_ssim = (validation_ssim) / len(validation_dataloader)
    return training_loss, validation_loss, training_psnr, validation_psnr, training_ssim, validation_ssim

if __name__ == '__main__':

    

    checkpoint_path = os.path.join('./stored_data', cfg.model_name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5)
        # transforms.RandomRotation((180))
    ])
    train_dataset = FujiDataset(cfg.root, cfg.train_preprocess_file_name, transform)
    validation_dataset = FujiDataset(cfg.root, cfg.val_preprocess_file_name, None)
    test_dataset = FujiDataset(cfg.root, cfg.test_file_name, None)
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=cfg.batch_size)
    validation_dataloader = DataLoader(validation_dataset, shuffle=True, batch_size=cfg.batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=cfg.batch_size)

    model = UNet().to(device)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999))
    model.train()
    train(model=model, train_dataloader=train_dataloader, validation_dataloader=validation_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device, epochs=cfg.epochs, checkpoint_path=checkpoint_path)

