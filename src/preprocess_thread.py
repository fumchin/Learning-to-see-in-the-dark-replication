import threading
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

def preprocess(i, thread_num, lock, f_lines, max_line):
    
    process_path = os.path.join(cfg.root, 'val_preprocess')

    process_short_path = os.path.join(process_path, 'short')
    process_long_path = os.path.join(process_path, 'long')
    
    # for index in tqdm(range(len(f_lines))):
    index = i
    while(index < max_line):
        
        print('index: ' + str(index))
        line = f_lines[index]
        str_list = line.split()
        src_img_path, target_img_path = str_list[0], str_list[1]
        src_img_path = os.path.join(root, src_img_path)
        target_img_path = os.path.join(root, target_img_path)
        
        src_file_name = os.path.splitext(os.path.basename(src_img_path))[0]
        target_file_name = os.path.splitext(os.path.basename(target_img_path))[0]
        if(os.path.exists(os.path.join(process_long_path, target_file_name)) == False):
            target_img = rawpy.imread(target_img_path) 
            target_rgb = target_img.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            target_rgb = target_rgb.astype(np.float32)
            np.save(os.path.join(process_long_path, target_file_name), target_rgb)
        
        if(os.path.exists(os.path.join(process_short_path, src_file_name)) == False):
            src_img = rawpy.imread(src_img_path)
            src_img = src_img.raw_image_visible.astype(np.float32)
            np.save(os.path.join(process_short_path, src_file_name), src_img)
    
        
        path_str = os.path.join(process_short_path, src_file_name) + ' ' + os.path.join(process_long_path, target_file_name) + '\n'
        lock.acquire()
        f= open(os.path.join(cfg.root, "val_preprocess_list.txt"),"a+")
        f.write(path_str)
        f.close()
        lock.release()
        
        index += thread_num
        

if __name__ == "__main__":

    root = cfg.root
    train_file_name = "Fuji_train_list.txt"
    test_file_name = "Fuji_test_list.txt"
    val_file_name = "Fuji_val_list.txt"
    process_path = os.path.join(root, 'val_preprocess')
    process_short_path = os.path.join(process_path, 'short')
    process_long_path = os.path.join(process_path, 'long')
    if not os.path.exists(os.path.join(process_path)):
        os.makedirs(process_path)
        os.makedirs(process_short_path)
        os.makedirs(process_long_path)
    
    # train preprocess
    f_lines = open(os.path.join(root, val_file_name), 'r').readlines()
    
    thread_num = 10
    max_line = len((f_lines))
    threads = []
    lock = threading.Lock()
    for i in range(thread_num):
        threads.append(threading.Thread(target = preprocess, args = (i, thread_num, lock, f_lines, max_line)))
        threads[i].start()
    
    for i in range(thread_num):
        threads[i].join()
    
        