from typing import Dict, Tuple
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
from torchvision.datasets import MNIST, Food101
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import json

from model import ResidualConvBlock, UnetDown, UnetUp, EmbedFC, ContextUnet, ddpm_schedules, DDPM

def generate(model_path, output_path, paras):
    ep = os.path.splitext(os.path.basename(model_path))[0].split('_')[-1]
    save_dir = os.path.join(output_path, f'outputs_gen/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    n_T = paras["n_T"]
    device = paras["device"]
    n_classes = paras["n_classes"]
    n_feat = paras["n_feat"]
    img_size = paras["img_size"]
    betas = paras["betas"]
    drop_p = paras["drop_p"]
    ws_test = paras["ws_test"]
    
    num_gen = 5
    # mean, std = torch.Tensor((0.485, 0.456, 0.406)).unsqueeze(1).unsqueeze(2), torch.Tensor((0.229, 0.224, 0.225)).unsqueeze(1).unsqueeze(2)
    invTrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes)
    ddpm = DDPM(nn_model, betas=betas, n_T=n_T, device=device, drop_prob=drop_p)
    ddpm.to(device)
    ddpm.load_state_dict(torch.load(model_path))
    ddpm.eval()
    with torch.no_grad():
        n_sample = num_gen*n_classes
        for w_i, w in enumerate(ws_test):
            x_gen, x_gen_store = ddpm.sample(n_sample, n_classes, (3, img_size, img_size), device, guide_w=w)
            # x_real = torch.Tensor(x_gen.shape).to(device)
            # trans_x_real = x_real*std.to(device) + mean.to(device)
            trans_x_gen = invTrans(x_gen)
            grid = make_grid(trans_x_gen, nrow=10)
            save_image(grid, save_dir + f"image{img_size}_ep{ep}.png")
            print('saved image at ' + save_dir + f"image{img_size}_ep{ep}.png")

if __name__ == "__main__":
    
    paras = {
      "datasource" : "food101",
      "n_epoch" : 200,
      "batch_size" : 100,
      "n_T" : 400,
      "device" : "cuda:0",
      "n_classes" : 101,
      "n_feat" : 256, # 128 ok, 256 better (but slower)
      "img_size" : 64,
      "lrate" : 5e-5,
      "betas" : (1e-4,0.02),
      "drop_p" : 0.2,
      "ws_test" : [0.5, 1.2, 2.0], # strength of generative guidance
    }
    
    # for epoch in (list(range(10, 200, 10)) + [{n_epoch}-1]):
    # model_path = f'/mnt/c/Code/cs771_project/data/outputs_64/model_39.pth'
    # output_path = '/mnt/c/Code/cs771_project/data/outputs_64'
    epoch = 60
    model_path = f'/storage08/shuchen/DDPM/outputs_w8_lr5/model_{epoch}.pth'
    output_path = '/storage08/shuchen/DDPM/outputs_12181930/'
    generate(model_path, output_path, paras)