import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

#### MY GAN CONFIGURATION FILE
import GAN_Configuration
####

def train_GAN_Model():
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(root=GAN_Configuration.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(GAN_Configuration.image_size),
                                   transforms.CenterCrop(GAN_Configuration.image_size),
                                   transforms.ToTensor(),
                                   transforms.Grayscale(),  # changed! to grayscale
                                   transforms.Normalize((0.5,), (0.5,)),  # changed to 1 channle!
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=GAN_Configuration.batch_size,
                                             shuffle=True, num_workers=GAN_Configuration.workers, pin_memory=True)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and GAN_Configuration.ngpu > 0) else "cpu")


    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()


    # custom weights initialization called on ``netG`` and ``netD``
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # Generator Code

    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d( GAN_Configuration.nz, GAN_Configuration.ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(GAN_Configuration.ngf * 8),
                nn.ReLU(True),  # changed from ReLU(True)!
                # state size. ``(ngf*8) x 4 x 4``
                nn.ConvTranspose2d(GAN_Configuration.ngf * 8, GAN_Configuration.ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(GAN_Configuration.ngf * 4),
                nn.ReLU(True),  # changed from ReLU(True)!
                # state size. ``(ngf*4) x 8 x 8``
                nn.ConvTranspose2d( GAN_Configuration.ngf * 4, GAN_Configuration.ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(GAN_Configuration.ngf * 2),
                nn.ReLU(True),  # changed from ReLU(True)!
                # state size. ``(ngf*2) x 16 x 16``
                nn.ConvTranspose2d( GAN_Configuration.ngf * 2, GAN_Configuration.ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(GAN_Configuration.ngf),
                nn.ReLU(True),  # changed from ReLU(True)!
                # state size. ``(ngf) x 32 x 32``
                nn.ConvTranspose2d(GAN_Configuration.ngf, GAN_Configuration.nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. ``(nc) x 64 x 64``
            )

        def forward(self, input):
            return self.main(input)

    # Create the generator
    netG = Generator(GAN_Configuration.ngpu).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (GAN_Configuration.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(GAN_Configuration.ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)

    # Print the model
    print(netG)

    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is ``(nc) x 64 x 64``
                nn.Conv2d(GAN_Configuration.nc, GAN_Configuration.ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf) x 32 x 32``
                nn.Conv2d(GAN_Configuration.ndf, GAN_Configuration.ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(GAN_Configuration.ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf*2) x 16 x 16``
                nn.Conv2d(GAN_Configuration.ndf * 2, GAN_Configuration.ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(GAN_Configuration.ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf*4) x 8 x 8``
                nn.Conv2d(GAN_Configuration.ndf * 4, GAN_Configuration.ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(GAN_Configuration.ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf*8) x 4 x 4``
                nn.Conv2d(GAN_Configuration.ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input)

    # Create the Discriminator
    netD = Discriminator(GAN_Configuration.ngpu).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (GAN_Configuration.ngpu > 1):
        netD = nn.DataParallel(netD, list(range(GAN_Configuration.ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)

    # Print the model
    print(netD)


    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, GAN_Configuration.nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=GAN_Configuration.lr, betas=(GAN_Configuration.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=GAN_Configuration.lr, betas=(GAN_Configuration.beta1, 0.999))


    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    Desired_loss_threshold = GAN_Configuration.Desired_loss_threshold  # Set your desired loss threshold here
    Percent_of_all_dataset_stop = GAN_Configuration.Percent_of_all_dataset_stop # Set percentage of the minimum TOTAL number of epoches to complete before stopping

    D_x = 0
    errD = 0
    errG = 0
    Last_loss_d = 0

    loss_g_lst =[]
    last_g_err_calculation =0

    #precentblack = BlackPrecentage.BlackPrec()
    #print(precentblack)


    print("Starting Training Loop...")
    # For each epoch
    countDx = 0
    accuracy = 0
    for epoch in range(GAN_Configuration.num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, GAN_Configuration.nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, GAN_Configuration.num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            countDx=countDx+1
            accuracy = accuracy+D_x

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            Last_loss_d = errD

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 20 == 0) or ((epoch == GAN_Configuration.num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

            loss_g_lst.append(errG.item())


        if epoch>=20:
            last_20_items = loss_g_lst[-20:]
            curr  = sum(last_20_items)/len(last_20_items)
            if abs(curr - Desired_loss_threshold) <= 1 and (epoch >= GAN_Configuration.num_epochs*Percent_of_all_dataset_stop) and D_x > 0.9:
                print(f"Losses are around {Desired_loss_threshold} for the Generator. Stopping training.")
                break

        '''abs(errG.item() - Desired_loss_threshold) <= 0.1 and (epoch >= GAN_Configuration.num_epochs*Percent_of_all_dataset_stop)'''
        '''
        # Check if both Loss_D and Loss_G are around the desired threshold
        if epoch == 500:
            print(f"Losses are around {Desired_loss_threshold} for the Generator. Stopping training.")
            break  # Exit the inner loop
        '''
    avgAcc = (accuracy / countDx) * 100
    print(f'Average accuracy for the learning process: {avgAcc:.3f}%')

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    ani.save('animation.gif', writer='pillow', fps=5)  # Adjust 'fps' as needed
    HTML(ani.to_jshtml())


    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()

    ############################################### SAVE THE MODEL

    # Save the generator and discriminator models
    torch.save(netG.state_dict(), GAN_Configuration.saved_models_path+'/generator.pth')
    torch.save(netD.state_dict(), GAN_Configuration.saved_models_path+'/discriminator.pth')

    # Save the optimizer states if needed
    torch.save(optimizerG.state_dict(), GAN_Configuration.saved_models_path+'/generator_optimizer.pth')
    torch.save(optimizerD.state_dict(), GAN_Configuration.saved_models_path+'/discriminator_optimizer.pth')
    #print(Last_loss_d.item())

