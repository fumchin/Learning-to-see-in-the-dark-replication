import os
import os.path
import rawpy
import numpy as np
from glob import glob
from PIL import Image
import imageio
import torch
import config as cfg

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

if __name__ == "__main__":

    root = cfg.root
    train_file_name = "Fuji_train_list.txt"
    test_file_name = "Fuji_test_list.txt"
    val_file_name = "Fuji_val_list.txt"
    train_process_path = os.path.join(root, 'train_preprocess')
    train_process_short_path = os.path.join(train_process_path, 'short')
    train_process_long_path = os.path.join(train_process_path, 'long')
    if not os.path.exists(os.path.join(train_process_path)):
        os.makedirs(train_process_path)
        os.makedirs(train_process_short_path)
        os.makedirs(train_process_long_path)
    
    
    # validation_process_path = os.path.join(root, 'val_preprocess')
    # validation_process_short_path = os.path.join(validation_process_path, 'short')
    # validation_process_long_path = os.path.join(validation_process_path, 'long')
    # if not os.path.exists(os.path.join(validation_process_path)):
    #     os.makedirs(validation_process_path)
    #     os.makedirs(validation_process_short_path)
    #     os.makedirs(validation_process_long_path)
    
    # train preprocess
    f_lines = open(os.path.join(root, train_file_name), 'r').readlines()
    f= open(os.path.join(cfg.root, "train_preprocess_list.txt"),"w")
    for index in tqdm(range(len(f_lines))):
        line = f_lines[index]
        str_list = line.split()
        src_img_path, target_img_path = str_list[0], str_list[1]
        src_img_path = os.path.join(root, src_img_path)
        target_img_path = os.path.join(root, target_img_path)
        
        src_file_name = os.path.splitext(os.path.basename(src_img_path))[0]
        target_file_name = os.path.splitext(os.path.basename(target_img_path))[0]
        
        src_img = rawpy.imread(src_img_path)
        target_img = rawpy.imread(target_img_path)

        src_img = src_img.raw_image_visible.astype(np.float32)
        target_rgb = target_img.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        target_rgb = target_rgb.astype(np.float32)

        np.save(os.path.join(train_process_short_path, src_file_name), src_img)
        np.save(os.path.join(train_process_long_path, target_file_name), target_rgb)
        path_str = os.path.join(train_process_short_path, src_file_name) + ' ' + os.path.join(train_process_long_path, target_file_name) + '\n'
        f.write(path_str)
    f.close()
        