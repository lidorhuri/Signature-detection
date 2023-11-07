import torch.cuda

import GAN_pythorch_train
import GAN_pythorch_prediction

print(torch.cuda.is_available())
# Train the GAN model
GAN_pythorch_train.train_GAN_Model()

# Upload the model
arr_model = GAN_pythorch_prediction.load_gan_model()
# Make samples and current sample-score to predict the outcome, passing the netG\D and noise
curr_GAN_score = GAN_pythorch_prediction.generate_samples_and_GAN_score(arr_model[0],arr_model[1],arr_model[2])

file_path = r"D:\works\SignProj\GAN-Model\CudaProject\conf.txt"
with open(file_path, 'w') as file:
    file.write(str(curr_GAN_score[1]))

print("the model is trained and ready for prediction! score: ",curr_GAN_score)