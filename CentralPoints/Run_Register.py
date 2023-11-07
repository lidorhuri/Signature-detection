import SVM_Model as svm
import Signtures_Images as si

def Register_Train():
    svm.train_model()
    si.Save_registred_signatures_image()
    #gan.train_model()
    return "Training done! You are allowed to connect"


print(Register_Train())