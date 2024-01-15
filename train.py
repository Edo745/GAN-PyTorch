import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import time
import os
from models.generator import Generator
from models.discriminator import Discriminator

def train():
    # loading the MNIST dataset
    dataset = dset.MNIST(root='GAN-PyTorch/data', download=True,
                        transform=transforms.Compose([
                            transforms.Resize(28),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)), #-1,1
                        ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    # checking the availability of cuda devices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    criterion = nn.BCELoss()
    
    # Initialize generator and discriminator
    netG = Generator().to(device) ; print(netG)
    netD = Discriminator().to(device) ; print(netD)

    # setup optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    nz = 100
    fixed_noise = torch.randn(64, nz, 1, 1, device=device) # input noise size: 100
    real_label = 1
    fake_label = 0

    nepochs = 5

    loss_g, loss_d, d_fake, d_real = [],[],[],[]

    print("\nTRAINING\n")
    
    for epoch in range(nepochs):
        for i, (data, labels) in enumerate(dataloader, 0):
            t_start = time.time()
        
            #------------------------------------------------------------
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            #------------------------------------------------------------
        
            optimizerD.zero_grad()

            # real samples
            real = data.to(device)
            batch_size = real.size(0) # 64
            label_r = torch.full((batch_size,), real_label, device=device).float() # 64 ones tensor

            # fake samples
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label_f = torch.full((batch_size,), fake_label, device=device).float() # 64 zeros tensor

            # create the batch
            batch = torch.cat((real, fake.detach())) # 128 size tensor containing 64 real images and 64 fake images
            labels =  torch.cat((label_r, label_f)) # 128 labels for the 64+64 real and fake images

            # Discriminator prediction
            output = netD(batch) # 128 probabilities
            
            # Binary Cross Entropy Loss
            errD = criterion(output, labels)
            errD.backward()

            # Mean response of D on real samples (to print, it has to increase)
            D_x = output[0:batch_size].mean().item()

            # Mean response of D on fake samples (to print, it has to decrease)
            D_G_z1 = output[batch_size:].mean().item() 

            real_cpu = data[0].to(device)

            # Updating the weights of the Discriminator
            optimizerD.step()

            #----------------------------------------------
            # (2) Update G network: minimize -log(D(G(z)))
            #----------------------------------------------
            optimizerG.zero_grad()

            # D(G(z)) 
            output = netD(fake) 

            # -log (D(G(z)))
            errG = -torch.log(output).mean() 
            errG.backward()

            # Mean response of on fake samples after the D update (to print, it has to increase)
            D_G_z2 = output.mean().item() 
            
            # Updating the weights of the Generator
            optimizerG.step() 
            
            # Monitoring boiler plate
            t_end = time.time()
            os.makedirs('GAN-Pytorch/results', exist_ok=True)
            if i % 100 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f %.1f images/s'
                        % (epoch+1, nepochs, i, len(dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, batch.shape[0]/(t_end-t_start)))

            if i % 100 == 0:
                vutils.save_image(real_cpu,'GAN-Pytorch/real_samples.png' ,normalize=True)
                fake = netG(fixed_noise)
                loss_g.append(errG.item())
                loss_d.append(errD.item())
                d_real.append(D_x)
                d_fake.append(D_G_z1)
                vutils.save_image(fake.detach(),'GAN-Pytorch/results/fake_samples_epoch_%03d.png' % (epoch), normalize=True)
        os.rmdir('GAN/checkpoints')
        os.makedirs('GAN/checkpoints', exist_ok=True)
        torch.save(netG.state_dict(), f'GAN/checkpoints/netG_epoch_{epoch + 1}.pth')
                
if __name__ == '__main__':
    train()
