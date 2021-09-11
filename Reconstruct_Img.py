import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import save_image
import pickle


# ----- Initialization ----- #
# Hyper Parameters
epoch = 199
testNum = 1
ImgSize = 32  # for resize
NodeNum = [3072, 1024, 512, 128, 64, 128, 512, 1024, 3072]  # the design same as training
# change directory
os.chdir(r"C:\...\Project")  # alter the directory !!!

FileName = './VAE/multi_0/1_199/vae.pkl'  # the VAE network saved !!!



# ----- Build VAE network ----- #
# NodeNum = [3072, 1024, 512, 128, 64, 128, 512, 1024, 3072]
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(NodeNum[0], NodeNum[1]),  # (3072, 1024)
            nn.ReLU(inplace=True),             # nn.Tanh(), # Activation func
            nn.Linear(NodeNum[1], NodeNum[2]),  # (1024, 512)
            nn.ReLU(inplace=True),
            nn.Linear(NodeNum[2], NodeNum[3]),  # (512, 128)
            nn.ReLU(inplace=True),
        )
        # mean vector
        self.muV = nn.Linear(NodeNum[3], NodeNum[4])  # (128, 64)
        # standard deviation vector
        self.varV = nn.Linear(NodeNum[3], NodeNum[4])  # (128, 64) # last layer could no activation func

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(NodeNum[4], NodeNum[5]),  # (64, 128)
            nn.ReLU(inplace=True),
            nn.Linear(NodeNum[5], NodeNum[6]),  # (128, 512)
            nn.ReLU(inplace=True),
            nn.Linear(NodeNum[6], NodeNum[7]),  # (512, 1024)
            nn.ReLU(inplace=True),
            nn.Linear(NodeNum[7], NodeNum[8]),  # (1024, 3072)
            nn.Sigmoid(),       # Activation function to bound the value between 0~1
        )
    def paraCombine(self, mul, varl):
        std = varl.mul(0.5).exp_()

        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()

        #eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mul)

    def forward(self, x):
        encoded = self.encoder(x)
        muLatent = self.muV(encoded)
        varLatent = self.varV(encoded)
        z = self.paraCombine(muLatent, varLatent)
        decoded = self.decoder(z)
        return decoded, muLatent, varLatent

vae = VAE()



# ----- Load VAE parameters ----- #
vae.load_state_dict(torch.load(FileName))
vae.eval()
Z_input = np.random.randn(64 ,NodeNum[4])  # mean=0, sigma=1
Input = torch.from_numpy(Z_input)

# transform from ndarray to torch
Input = Input.type(torch.FloatTensor)



# ----- generate fake images with trained model ----- #
decodedELE = vae.decoder(Input)

Img_Size = 3072

Output = decodedELE.data
img_0 = Output[:, np.arange(0, Img_Size, 3)]
img_1 = Output[:, np.arange(1, Img_Size, 3)]
img_2 = Output[:, np.arange(2, Img_Size, 3)]


def Transorm_to_Img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, ImgSize, ImgSize)
    return x


storeImg_0 = Transorm_to_Img(img_0)
storeImg_1 = Transorm_to_Img(img_1)
storeImg_2 = Transorm_to_Img(img_2)
storeImg = torch.cat((storeImg_2, storeImg_1), 1)
storeImg = torch.cat((storeImg, storeImg_0), 1)
save_image(storeImg, ('./VAE/Synth_{}.png'.format(epoch)))