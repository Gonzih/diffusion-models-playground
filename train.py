import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import matplotlib.pyplot as plt
import numpy as np

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1'            # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    './images-ready',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True                        # turn on mixed precision
)

trainer.train()

def show_image(image):
  plt.imshow(image.permute(1, 2, 0))

def show_images(images, nrows=4, ncols=4):
  h, w = 10, 10        # for raster image
  figsize = [8, 8]     # figure size, inches

  fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

  for i, axi in enumerate(ax.flat):
    # i runs from 0 to (nrows*ncols-1)
    # axi is equivalent with ax[rowid][colid]
    img = images[i]
    axi.imshow(img.permute(1, 2, 0))
    # get indices of row/column
    rowid = i // ncols
    colid = i % ncols
    # write row/col indices as axes' title for identification
    axi.set_title("Row:"+str(rowid)+", Col:"+str(colid))

  plt.tight_layout(True)
  plt.show()

sampled_images = diffusion.sample(batch_size = 16)
show_images(sampled_images.cpu(), 4, 4)
