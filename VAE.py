import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import os



# ----- Initialization ----- #
# Hyper Parameters
testNum = 1
EPOCH = 500
BATCH_SIZE = 32
LR = 0.001
ImgSize = 32  # for resize
NodeNum = [3072, 1024, 512, 128, 64, 128, 512, 1024, 3072]  # model design
MulTerm = 0

# change directory
os.chdir(r"C:\...\Project")  # alter the directory !!!

# Load Data
FilePath = "./Processed_Img/"
ImgFile = "Img_merge_32.npy"

# load Data & label
Img = np.load(FilePath + ImgFile)  # [21551 X 3072]
picture_Num = Img.shape[0]
Img_Size = Img.shape[1]

# construct directory to store result
if not os.path.exists('./VAE'):
    os.mkdir('./VAE')



# ----- dataset for VAE network ----- #
# Create Dataset
class TorchDataset(Data.Dataset):
    def __init__(self, Img_Data, Img_Label):
        """
        :param Img_Label: Img_Data
        :param len: picture_Num
        """
        self.Label = Img_Label
        self.len = len(self.Label)
        self.Data = Img_Data

        '''class torchvision.transforms.ToTensor'''
        self.toTensor = transforms.ToTensor()

    def __getitem__(self, i):
        index = i % self.len
        label = self.Label[index, :]  # label --> [1 X 3072]
        # Data  --> [1 X 3072]
        # transform from ndarray to torch
        Input = torch.from_numpy(self.Data[index, :])
        # transform from ndarray to torch
        Input = Input.type(torch.FloatTensor)
        return Input, label

    def __len__(self):
        return self.len



# ----- dataloader for VAE network ----- #
train_data = TorchDataset(Img, Img)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = TorchDataset(Img[:64, :], Img[:64, :])
test_loader = Data.DataLoader(dataset=test_data, batch_size=64, shuffle=False)



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

#  Optional to use GPU
if torch.cuda.is_available():
    vae.cuda()



# ----- function & parameter for Traing ----- #
optimizer = torch.optim.Adam(vae.parameters(), lr=LR)

reconstruc_function = nn.MSELoss(size_average=False) # ==> return loss.sum()
# a. if size_average = True，return loss.mean()
# b. if size_average = False，return loss.sum()
# default : reduce = True，size_average = True

def loss_function(recon_x, x, mu, logvar):
    # variational auto-encoder have to "self-define KL divergence"
    BCE = reconstruc_function(recon_x, x)  # mse loss
    KLD = (-0.5)*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # KL divergence
    return BCE + MulTerm*KLD, KLD


def Transorm_to_Img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, ImgSize, ImgSize)
    return x



# ----- Traing & Testing ----- #
train_loss = []
KL = []
for epoch in range(EPOCH):
    vae.train()
    this_loss = 0
    KL_D = 0
    for step, (x, y) in enumerate(train_loader):
        # x is same as y
        if torch.cuda.is_available():
            # label --> [1 X 3072]
            b_x = x.view(-1, Img_Size).cuda()
            b_y = x.view(-1, Img_Size).cuda()

        decodedELE, muL, varL = vae(b_x)
        loss, KL_value = loss_function(decodedELE, b_y, muL, varL)
        optimizer.zero_grad()
        loss.backward()
        this_loss += loss.data
        KL_D += KL_value
        optimizer.step()
        
    train_loss.append(float(this_loss/picture_Num))
    KL.append(float(KL_D/picture_Num))

    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, float(this_loss/picture_Num)))
    print('Train Epoch: {} \tKL_D: {:.6f}'.format(epoch, float(KL_D/picture_Num)))

    if (epoch+1)%50 == 0:
        # construct directory to store result
        NewFileDir = './VAE/' + str(testNum) + '_' + str(epoch)
        if not os.path.exists(NewFileDir):
            os.mkdir(NewFileDir)
        torch.save(vae.state_dict(), NewFileDir + "/vae.pkl")

        for step, (x, y) in enumerate(test_loader):
            # x is same as y
            if torch.cuda.is_available():
                #img = img.cuda()
                b_x2 = x.view(-1, Img_Size).cuda()
                b_y2 = x.view(-1, Img_Size).cuda()
            decodedELE2, muL2, varL2 = vae(b_x2)
            Output = decodedELE2.cpu().data
            img_0 = Output[:,np.arange(0, Img_Size, 3)]
            img_1 = Output[:,np.arange(1, Img_Size, 3)]
            img_2 = Output[:,np.arange(2, Img_Size, 3)]

            storeImg_0 = Transorm_to_Img(img_0)
            storeImg_1 = Transorm_to_Img(img_1)
            storeImg_2 = Transorm_to_Img(img_2)
            storeImg = torch.cat((storeImg_2, storeImg_1), 1)
            storeImg = torch.cat((storeImg, storeImg_0), 1)
            save_image(storeImg, (NewFileDir + '/IMG_{}.png'.format(epoch)) )
            


# ---- Restore pixels to 2D Images ----- #
Org = torch.from_numpy(Img[:64, :])
img_0 = Org[:, np.arange(0, Img_Size, 3)]
img_1 = Org[:, np.arange(1, Img_Size, 3)]
img_2 = Org[:, np.arange(2, Img_Size, 3)]

storeImg_0 = Transorm_to_Img(img_0)
storeImg_1 = Transorm_to_Img(img_1)
storeImg_2 = Transorm_to_Img(img_2)
storeImg = torch.cat((storeImg_2, storeImg_1), 1)
storeImg = torch.cat((storeImg, storeImg_0), 1)
save_image(storeImg, './VAE/Org_image.png')



# ---- Store Results ----- #
np.save("./VAE/train_loss.npy", train_loss)
np.save("./VAE/KLD.npy", KL)



# ---- Plot Images ----- #
plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
plt.plot(train_loss, color="blue", linewidth=1, label="$TrainLoss$")
plt.xlabel("Epoch(s)")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(KL, color="red", linewidth=1, label="$KLD$")
plt.xlabel("Epoch(s)")
plt.ylabel("KLD")
plt.legend()

plt.show()




