import torch
import constants.constants as constants
from models.GenerateClass import Generate
from torch_dataset import TrainSet
from utils.noise_generator import NoiseGenerator

class GenerateModel1(Generate):
    def __init__(self):
        super(GenerateModel1, self).__init__()

    def generate_motion(self, model, prosody):
        prosody = self.reshape_prosody(prosody)

        noise_g = NoiseGenerator(self.device)
        noise = noise_g.getNoise(constants.noise_size, prosody.shape[0]) 

        with torch.no_grad():
            output_eye, output_pose_r, output_au = model.forward(prosody, noise)
        outs = self.reshape_output(output_eye, output_pose_r, output_au)
        return outs

