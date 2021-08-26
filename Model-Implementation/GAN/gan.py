import torch 
import torch.nn
import torch.utils.data 
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import matplotlib.animation as animation
from IPython.display import HTML



#data directory and jpg/png image files should inside two folders
dataroot = './gan_image'

batch_size = 128
image_size = 64
nz = 100
num_epochs = 5
lr = 0.0002
beta1=0.5




dataset = dset.ImageFolder(root = dataroot,
                           transform = transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                          ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

real_batch = next(iter(dataloader))  ##

plt.figure(figsize=(8,16))
plt.axis("off")
plt.title("training dataset")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].cuda(), padding=2, normalize=True).cpu()))

#Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 1024, 4, 1, 0),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            #(4x4x1024)
            
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
              
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
                             
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
                              
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
            
            # ( 3x 64 x 64)
                   
        )
        
        
    def forward(self, input):  # use when apply
        return self.main(input)
    

netG = Generator().cuda()
print(netG)

#Discriminator

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__(),
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            #

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),


            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1 ,0),
            nn.Sigmoid()
            # (1)

        )


    def forward(self, input):  # use when apply
        return self.main(input)

netD = Discriminator().cuda()
print(netD)

# weight initialize
#netG = 

def weight_init(m):
    classname = m.__class__.__name__
    # print(classname.find('Conv')) if it's conv layer it prints 0, if not, prints -1
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        

netG.apply(weight_init)
netD.apply(weight_init)


criterion = nn.BCELoss()
z = torch.randn(batch_size, nz, 1, 1).cuda()

optimizerD = optim.Adam(netD.parameters(), lr = lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = lr, betas=(beta1, 0.999))

G_losses = []
D_losses = []

iters = 0
img_list = []

real_label = 1
fake_label = 0

for epoch in range(num_epochs):
    # batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        # 1. maximize D
        #train discriminator
        netD.zero_grad()
        b = data[0].size(0)
        label = torch.full((b,), real_label, dtype= torch.float).cuda()
        output = netD(data[0].cuda()).view(-1)
        errD_real = criterion(output, label)  # loss for logD(x)
        errD_real.backward()

        #D(G(z))
        z = torch.randn(b, nz, 1, 1).cuda()
        fake = netG(z)
        label.fill_(fake_label) # changed to 1 -> 0
        output = netD(fake.detach()).view(-1) # for not to train generator
        errD_fake = criterion(output, label)
        errD_fake.backward()

        errD = errD_real + errD_fake
        optimizerD.step()    # step = train!

        #train generator
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        if i%50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f \t \tLoss_G: %.4f'
                     % (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item()))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (iters % 500 == 0) or ((epoch==num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(z).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters+=1

## show training scalars to plt
plt.figure(figsize=(10,5))
plt.title("GEnerator and Discriminator Loss DUring Training")
plt.plt(G_losses, label='G')
plt.plt(D_loosses, label='D')
plt.xlabel("iterations")
plt.ylabel("loss")
plt.legend()
plt.imshow()

## animation result

#import matplotlib.animation as animation
#from IPython.display import HTML

fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay = 1000, blit=True)

HTML(ani.to_jshtml())

