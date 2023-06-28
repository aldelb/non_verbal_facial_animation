import argparse
from genericpath import isdir
import os
import pickle
import numpy as np
import torch
import constants.constants as constants
from constants.constants_utils import read_params
from utils.create_final_file import createFinalFile
from utils.model_utils import find_model, load_model
import pandas as pd
from os.path import join

gaze_columns = ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y"]
translation_columns = ["pose_Tx", "pose_Ty", "pose_Tz"]
au_columns = ["pose_Rx", "pose_Ry", "pose_Rz", "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-params', help='Path to the constant file', default="./params.cfg")
    parser.add_argument('-epoch', help='number of epoch of recovred model', default=9)
    parser.add_argument('-file', help='wich file name in default path (see code for default path)', default="")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parser.parse_args()
    read_params(args.params, "generate")

    model_file = find_model(int(args.epoch)) 
    model = load_model(constants.saved_path + model_file, device)

    path_data_out = constants.output_path + model_file[0:-3] + "/"
    if(not isdir(path_data_out)):
        os.makedirs(path_data_out, exist_ok=True)

    path = "/gpfsdswork/projects/rech/urk/uln35en/Data/audio_file/"
    set_dir = "set/"
    set_file = join(path, set_dir, args.file+".p")
    set_interval_file = join(path, set_dir, args.file+"_interval.p")

    with open(set_file, 'rb') as f:
        x = pickle.load(f)
    X = np.array(x)[:,:,np.r_[constants.selected_os_index_columns]]

    with open(set_interval_file, 'rb') as f:
        current_interval = pickle.load(f)

    columns = constants.openface_columns


    df_list = []
    print("Generation of video", args.file, "...")
    for index, data in enumerate(X, 0):
        input = data
        key = current_interval[index][0]
        out = constants.generate_motion(model, input)
        #add timestamp and head translation for greta
        timestamp = np.array(current_interval[index][1][:,0])
        out = np.concatenate((timestamp.reshape(-1,1), out[:,:constants.eye_size], np.zeros((out.shape[0], 3)), out[:,constants.eye_size:]), axis=1)
        df = pd.DataFrame(data = out, columns = columns)
        df_list.append(df)

    createFinalFile(path_data_out, key, df_list)