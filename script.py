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

from model import ResidualConvBlock, UnetDown, UnetUp, EmbedFC, ContextUnet, ddpm_schedules, DDPM

def train_mnist(data_path, save_dir):
    # hardcoding these here
    n_epoch = 20
    batch_size = 100
    n_T = 400 # 500
    device = "cuda:0"
    # device = "cpu"
    n_classes = 101
    n_feat = 128 # 128 ok, 256 better (but slower)
    img_size = 32
    lrate = 1e-4
    save_model = True
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ws_test = [2.0] # strength of generative guidance

    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))
    # tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1
    
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    invTrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                    std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                            transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                    std = [ 1., 1., 1. ]),
                            ])
    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    dataset = datasets.ImageFolder(root=data_path, transform=train_transform)
    # dataset = Food101(data_path, split='train', download=True, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.AdamW(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        with torch.no_grad():
            n_sample = 1*n_classes
            for w_i, w in enumerate(ws_test):
                x_gen, x_gen_store = ddpm.sample(n_sample, n_classes, (3, img_size, img_size), device, guide_w=w)
                trans_x_gen = invTrans(x_gen)
                grid = make_grid(trans_x_gen, nrow=10)
                save_image(grid, save_dir + f"image_ep{ep}_w{w}_norm.png")
                print('saved image at ' + save_dir + f"image_ep{ep}_w{w}_norm.png")

                # append some real images at bottom, order by class also
                # x_real = torch.Tensor(x_gen.shape).to(device)
                # for k in range(n_classes):
                #     for j in range(int(n_sample/n_classes)):
                #         try: 
                #             idx = torch.squeeze((c == k).nonzero())[j]
                #         except:
                #             idx = 0
                #         x_real[k+(j*n_classes)] = x[idx]

                # x_all = torch.cat([x_gen, x_real])
                # grid = make_grid(x_all*-1 + 1, nrow=10)
                # save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
                # print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")
                
                # if ep%10==0 or ep == int(n_epoch-1):
                #     # create gif of images evolving over time, based on x_gen_store
                #     fig, axs = plt.subplots(nrows=int(n_sample/n_classes), ncols=n_classes,sharex=True,sharey=True,figsize=(8,3))
                #     def animate_diff(i, x_gen_store):
                #         print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                #         plots = []
                #         for row in range(int(n_sample/n_classes)):
                #             for col in range(n_classes):
                #                 axs[row, col].clear()
                #                 axs[row, col].set_xticks([])
                #                 axs[row, col].set_yticks([])
                #                 # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                #                 plots.append(axs[row, col].imshow(-x_gen_store[i,(row*n_classes)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                #         return plots
                #     ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
                #     ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
                #     print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
        # optionally save model
        if save_model and (ep%5 == 0 or ep == int(n_epoch-1)):
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")

def generate(model_path, output_path):
    ep = os.path.splitext(os.path.basename(model_path))[0].split('_')[-1]
    save_dir = os.path.join(output_path, 'outputs_gen/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_feat = 128
    n_classes = 101
    n_T = 400
    img_size = 64
    device = "cuda:0"
    ws_test = [2.0]
    # mean, std = torch.Tensor((0.485, 0.456, 0.406)).unsqueeze(1).unsqueeze(2), torch.Tensor((0.229, 0.224, 0.225)).unsqueeze(1).unsqueeze(2)
    invTrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes)
    ddpm = DDPM(nn_model, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)
    ddpm.load_state_dict(torch.load(model_path))
    ddpm.eval()
    with torch.no_grad():
        n_sample = 1*n_classes
        for w_i, w in enumerate(ws_test):
            x_gen, x_gen_store = ddpm.sample(n_sample, n_classes, (3, img_size, img_size), device, guide_w=w)
            # x_real = torch.Tensor(x_gen.shape).to(device)
            # trans_x_real = x_real*std.to(device) + mean.to(device)
            trans_x_gen = invTrans(x_gen)
            grid = make_grid(trans_x_gen, nrow=10)
            save_image(grid, save_dir + f"image{img_size}_ep{ep}_w{w}_norm.png")
            print('saved image at ' + save_dir + f"image{img_size}_ep{ep}_w{w}_norm.png")


if __name__ == "__main__":
    data_path = '/mnt/c/Code/cs771_project/data/food-101-subset/images'
    save_dir = '/mnt/c/Code/cs771_project/data/outputs_class5'
    # data_path = '/storage08/shuchen/DDPM/'
    train_mnist(data_path, save_dir)

    model_path = f'/mnt/c/Code/cs771_project/data/outputs_64/model_39.pth'
    output_path = '/mnt/c/Code/cs771_project/data/outputs_64'
    # generate(model_path, output_path)

