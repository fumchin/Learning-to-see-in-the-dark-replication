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


class FujiDataset(Dataset):
    def __init__(self, root, file_name, transform):
        self.root = root
        self.file_name = file_name
        self.transform = transform
        self.f_lines = open(os.path.join(self.root, self.file_name), 'r').readlines()
        
    def __getitem__(self, index):
        line = self.f_lines[index]
        str_list = line.split()
        src_img_path, target_img_path = str_list[0], str_list[1]
        src_img_path = os.path.join(self.root, src_img_path)
        target_img_path = os.path.join(self.root, target_img_path)
        
        # src_img = rawpy.imread(src_img_path)
        # target_img = rawpy.imread(target_img_path)

        # src_img = src_img.raw_image_visible.astype(np.float32)
        # src_img = np.maximum(src_img - 1024, 0) / (16383 - 1024)
        # target_rgb = target_img.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        # target_rgb = np.expand_dims(np.float32(target_rgb / 65535.0), axis=0)
        # target_rgb = target_rgb.raw_image_visible.astype(np.float32)
        # target_rgb = Image.fromarray((target_rgb))
        # imageio.imwrite('./result_2.png',target_rgb) 
        # target_rgb.save(os.path.join("./apple.png"))
        src_img = np.load(src_img_path+".npy")
        target_rgb = np.load(target_img_path+".npy")

        H = (src_img.shape[0] // 6) * 6
        W = (src_img.shape[1] // 6) * 6

        start_H = np.random.randint(0, H - cfg.patch_size)
        start_W = np.random.randint(0, W - cfg.patch_size)
        
        src_img = src_img[start_H:start_H+cfg.patch_size, start_W:start_W+cfg.patch_size]
        src_pack = self.pack(src_img)
        src_pack = torch.tensor(src_pack, dtype=torch.float32)

        # target_img = target_img[0:cfg.patch_size, 0:cfg.patch_size]

        
        target_rgb = target_rgb[start_H:start_H+cfg.patch_size, start_W:start_W+cfg.patch_size]
        # target_rgb = Image.fromarray((target_rgb*255).astype(np.uint16))
        # src_img = Image.fromarray((src_img))
        # target_rgb.save(os.path.join("./apple.png"))
        target_rgb = torch.tensor(target_rgb)
        
        target_rgb = target_rgb.permute(2, 0, 1)
        src_pack = src_pack.permute(2, 0, 1)

        

        return src_pack, target_rgb

        pass
        
    def __len__(self):
        total_size = len(self.f_lines)
        return total_size
    
    def pack(self, im):
        
        im = np.maximum(im - 1024, 0) / (16383 - 1024)  # subtract the black level

        img_shape = im.shape
        H = (img_shape[0] // 6) * 6
        W = (img_shape[1] // 6) * 6

        out = np.zeros((H // 3, W // 3, 9))

        # 0 R
        a = out[0::2, 0::2, 0]
        b = im[0:H:6, 0:W:6]
        out[0::2, 0::2, 0] = im[0:H:6, 0:W:6]
        out[0::2, 1::2, 0] = im[0:H:6, 4:W:6]
        out[1::2, 0::2, 0] = im[3:H:6, 1:W:6]
        out[1::2, 1::2, 0] = im[3:H:6, 3:W:6]

        # 1 G
        out[0::2, 0::2, 1] = im[0:H:6, 2:W:6]
        out[0::2, 1::2, 1] = im[0:H:6, 5:W:6]
        out[1::2, 0::2, 1] = im[3:H:6, 2:W:6]
        out[1::2, 1::2, 1] = im[3:H:6, 5:W:6]

        # 1 B
        out[0::2, 0::2, 2] = im[0:H:6, 1:W:6]
        out[0::2, 1::2, 2] = im[0:H:6, 3:W:6]
        out[1::2, 0::2, 2] = im[3:H:6, 0:W:6]
        out[1::2, 1::2, 2] = im[3:H:6, 4:W:6]

        # 4 R
        out[0::2, 0::2, 3] = im[1:H:6, 2:W:6]
        out[0::2, 1::2, 3] = im[2:H:6, 5:W:6]
        out[1::2, 0::2, 3] = im[5:H:6, 2:W:6]
        out[1::2, 1::2, 3] = im[4:H:6, 5:W:6]

        # 5 B
        out[0::2, 0::2, 4] = im[2:H:6, 2:W:6]
        out[0::2, 1::2, 4] = im[1:H:6, 5:W:6]
        out[1::2, 0::2, 4] = im[4:H:6, 2:W:6]
        out[1::2, 1::2, 4] = im[5:H:6, 5:W:6]

        out[:, :, 5] = im[1:H:3, 0:W:3]
        out[:, :, 6] = im[1:H:3, 1:W:3]
        out[:, :, 7] = im[2:H:3, 0:W:3]
        out[:, :, 8] = im[2:H:3, 1:W:3]
        return out
    
    

if __name__ == "__main__":
    file_name = "Fuji_train_list.txt"
    root = "/home/fumchin/data/cv/final/dataset"
    fuji = FujiDataset(root, file_name, None)
    train_dataloader = DataLoader(fuji, batch_size=24, shuffle=False)
    a, b = next(iter(train_dataloader))
    print(a, b)
    print('f')