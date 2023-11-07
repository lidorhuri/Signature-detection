import os
import torch
from D_G_Classes import Generator, Discriminator
import torchvision.utils as vutils
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

#config file!
import GAN_Configuration
# using conda env "myenv" with pythorch and opencv

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and GAN_Configuration.ngpu > 0) else "cpu")

# global vars to use
G_netG = None
G_netD = None
G_noise = None
output_dir = GAN_Configuration.GAN_generated_samples_path

preprocess = transforms.Compose([
                   transforms.Resize(GAN_Configuration.image_size),
                   transforms.CenterCrop(GAN_Configuration.image_size),
                   transforms.ToTensor(),
                   transforms.Grayscale(),  # changed! to grayscale
                   transforms.Normalize((0.5,), (0.5,)),  # changed to 1 channle!
    ])


# Define the paths to the saved models for generator and discriminator
model_dir = GAN_Configuration.saved_models_path
generator_path = os.path.join(model_dir, 'generator.pth')
discriminator_path = os.path.join(model_dir, 'discriminator.pth')

def load_gan_model():
    # Define the paths to the saved models for generator and discriminator
    model_dir = GAN_Configuration.saved_models_path
    generator_path = os.path.join(model_dir, 'generator.pth')
    discriminator_path = os.path.join(model_dir, 'discriminator.pth')

    # Check if the directory and model files exist
    if not os.path.exists(model_dir) or not os.path.isfile(generator_path) or not os.path.isfile(discriminator_path):
        print("Model files do not exist. Please train and save the models first.")
        exit()


    # Create the generator and discriminator
    netG = Generator(GAN_Configuration.ngpu).to(device)
    netD = Discriminator(GAN_Configuration.ngpu).to(device)

    # Load the trained generator and discriminator models
    netG.load_state_dict(torch.load(generator_path))
    netD.load_state_dict(torch.load(discriminator_path))

    # Set both models to evaluation mode
    netG.eval()
    netD.eval()

    # Generate random noise (z) for generating new samples
    noise = torch.randn(GAN_Configuration.batch_size, GAN_Configuration.nz, 1, 1, device=device)

    # Generate fake samples using the generator
    with torch.no_grad():
        fake_samples = netG(noise)

    # Pass the generated samples through the discriminator
    with torch.no_grad():
        discriminator_output = netD(fake_samples)


    os.makedirs(output_dir, exist_ok=True)

    # Generate random noise (z) for generating new samples ## GETTING A NEW NOISE SETTINGS FOR 10 IMAGES
    noise = torch.randn(10, GAN_Configuration.nz, 1, 1, device=device)  # 10 samples

    return [netG , netD , noise]

def generate_samples_and_GAN_score(G_netG , G_netD , G_noise):
    ############ getting 10 image samples
    best_discriminator_output_real_gen = 100
    discriminator_output_real_gen = 0
    scores = []
    # Generate and save individual samples
    for i in range(10):
        with torch.no_grad():
            fake_sample = G_netG(G_noise[i:i+1]).to(device)  # Generate one sample at a time

        # Convert the sample to a NumPy array and transpose it
        fake_sample_numpy = np.transpose(fake_sample.cpu().squeeze().numpy())

        # Save the sample as an image
        filename = os.path.join(output_dir, f"generated_sample_{i}.png")
        vutils.save_image(fake_sample, filename, normalize=True)


    ############# prediction of the generator's pictures (best of 10)
    sum_score=0
    # Define a transform to preprocess the new image
    for i in range(10):
    # the login image at the GAN root directory
        new_generator_path = GAN_Configuration.GAN_generated_samples_path+f'/generated_sample_{i}.png'

        # Load and preprocess the new image
        new_generator_path = Image.open(new_generator_path).convert('L')  # Convert to grayscale
        new_generator_path = preprocess(new_generator_path).unsqueeze(0)  # Add batch dimension
        new_generator_path = new_generator_path.to(device)
      

        with torch.no_grad():
            discriminator_output_real_gen = G_netD(new_generator_path)


        sum_score += discriminator_output_real_gen.item()
        scores.append(discriminator_output_real_gen.item())

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(mean_score, std_score)
    return mean_score, std_score

   # sum_score = sum_score/10
    #print(sum_score)
    #return sum_score

    #print(best_discriminator_output_real_gen.item())
    #return best_discriminator_output_real_gen


def predict_login(mean_discriminator_output_real_gen, std_discriminator_output_real_gen):
    ################################################################################################################### prediction
    # the login image at the GAN root directory
    new_image_path = GAN_Configuration.GAN_project_path+'/login.png'

    # Load and preprocess the new image
    new_image = Image.open(new_image_path).convert('L')  # Convert to grayscale
    new_image = preprocess(new_image).unsqueeze(0)  # Add batch dimension
    new_image = new_image.to(device)
     
    filename = os.path.join(output_dir, f"generated_sample_login.png")
    vutils.save_image(new_image, filename, normalize=True)

    # Create the generator and discriminator
    netD = Discriminator(GAN_Configuration.ngpu).to(device)
    # Load the trained generator and discriminator models
    netD.load_state_dict(torch.load(discriminator_path))
    # Set both models to evaluation mode
    netD.eval()

    # Pass the new image through the discriminator
    with torch.no_grad():
        discriminator_output_real = netD(new_image)

        # Set a range around the mean using standard deviation
    lower_bound = (mean_discriminator_output_real_gen - std_discriminator_output_real_gen)  / (100000)
    upper_bound = (mean_discriminator_output_real_gen + std_discriminator_output_real_gen)* (100000)

    # Check if the score falls within the range
    is_real = lower_bound <= discriminator_output_real <= upper_bound

    # Calculate accuracy
    if is_real:
        return True
    else:
        return False





