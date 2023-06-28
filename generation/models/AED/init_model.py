import constants.constants as constants
from models.AED.generating import GenerateModel1
from models.AED.model import Generator as model1
from models.AED.w_training import TrainModel1_W
from models.AED.training import TrainModel1



def init_model_1(task):
    if(task == "train"):
        if(constants.w_gan == True):
            print("WGAN MODEL")
            train = TrainModel1_W(gan=True)
            constants.train_model = train.train_model_w
        else:
            print("Unroll model")
            train = TrainModel1(gan=True)
            constants.train_model = train.train_model
    elif(task == "generate"):
        constants.model = model1
        generator = GenerateModel1()
        constants.generate_motion = generator.generate_motion