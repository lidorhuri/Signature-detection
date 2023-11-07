import RaisingHand as rh
import Direction as dir
import SVM_Model as svm
import Signtures_Images as si
import subprocess

def run_gan_prediction():
    program_path = "D:\\works\\SignProj\\GAN-Model\\CudaProject\\GAN_pythorch_prediction.py"
    subprocess.run(["python", program_path], capture_output=True, text=True)
    file_path = "D:\\works\\SignProj\\GAN-Model\\CudaProject\\result.txt"
    with open(file_path, 'r') as file:
        result = file.read().strip()  # Read the file content and remove any leading/trailing whitespaces
    return result

def Login_Test():
    si.Save_login_signture_image()

    #layer 1: Raising Hand count
    result = rh.Comparing_write_breaks()
    if (result==False):
        return "Error in the number of times you raised your signature confusing hand"


    #layer 2: Direction
    result = dir.activate_direction()
    if (result==False):
        return "An error occurred in one of the signature directions"

    #layer 3: SVM
    result = svm.load_model()
    if (result == False):
       return "An error occurred in the signature structure (SVM)"

    #layer 4: GAN
    result = run_gan_prediction()
    if(result == 'False'):
        return "An error occurred in the identification of the signature (GAN)"



    #Identification successful:
    if (result):
       return "The connection was made successfully"

print(Login_Test())


