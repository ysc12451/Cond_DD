import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from torchvision.datasets import MNIST, Food101
from torchvision.utils import save_image, make_grid
from PIL import Image, ImageDraw, ImageOps

cls = "pizza"
data_path = f'/mnt/c/Code/cs771_project/data/food-101-subset/images/{cls}'
save_dir = os.path.join(data_path, 'normalized/')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
ws_test = [2.0] # strength of generative guidance
imag_size = 64
batch_size = 32
raw_transform = transforms.Compose([
    transforms.Resize(imag_size),
    transforms.CenterCrop(imag_size),
    transforms.ToTensor()])

train_transform = transforms.Compose([
    transforms.Resize(imag_size),
    transforms.CenterCrop(imag_size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


original_image = Image.open(os.path.join(data_path, "22489.jpg"))

# Apply a transformation (for example, invert colors)
raw_image = raw_transform(original_image)
norm_image = train_transform(original_image)
raw_image = transforms.ToPILImage()(raw_image)
norm_image = transforms.ToPILImage()(norm_image)


raw_path = os.path.join(save_dir, f"{cls}_raw.png")  # Replace with the desired output path
norm_path = os.path.join(save_dir, f"{cls}_norm.png")  # Replace with the desired output path
raw_image.save(raw_path)
norm_image.save(norm_path)