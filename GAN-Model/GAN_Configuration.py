import random
import torch


manualSeed = 9999
#manualSeed = random.randint(1, 10000) # use if you want new results



#print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

GAN_generated_samples_path = r'D:\works\SignProj\GAN-Model\CudaProject\GAN_generated_samples'
saved_models_path = r'D:\works\SignProj\GAN-Model\CudaProject\saved_models'
GAN_project_path =  r'D:\works\SignProj\GAN-Model\CudaProject'

# Root directory for dataset
dataroot = r'D:\works\SignProj\GAN-Model\CudaProject\image_dataset'
#"./image_dataset"

# Number of workers for dataloader
workers = 0

# Batch size during training
batch_size = 200

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 6

# Size of feature maps in generator
ngf = 75

# Size of feature maps in discriminator
ndf = 75

# Number of training epochs
num_epochs = 2000

# Learning rate for optimizers0.0008/
lr=0.0005



# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


Desired_loss_threshold = 4
Percent_of_all_dataset_stop = 0.25