import torch
import constants.constants as constants
from torch_dataset import TrainSet

class Generate():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("*"*10, "cuda available ", torch.cuda.is_available(), "*"*10)
        self.dset = TrainSet()

    def reshape_prosody(self, input):
        input = self.dset.scale_pros(input)
        input = torch.tensor(input, device = self.device).float()
        return input
    
    def reshape_pose(self, input):
        input = self.dset.scale_pose(input)
        input = torch.tensor(input, device = self.device).float()
        return input

    def separate_openface_features(self, input):
        input_eye = torch.index_select(input, dim=2, index = torch.tensor(range(constants.eye_size), device=self.device))
        input_pose_r = torch.index_select(input, dim=2, index=torch.tensor(range(constants.eye_size, constants.eye_size + constants.pose_r_size), device=self.device))
        input_au = torch.index_select(input, dim=2, index=torch.tensor(range(constants.pose_size, constants.pose_size + constants.au_size), device=self.device))
        return input_eye, input_pose_r, input_au
    
    def reshape_output(self, output_eye, output_pose_r, output_au):
        outs = torch.cat((output_eye, output_pose_r, output_au), 2)
        outs = self.dset.rescale_pose(outs.cpu())
        return outs
    
    def reshape_single_output(self, output):
        outs = self.dset.rescale_pose(output.cpu())
        return outs