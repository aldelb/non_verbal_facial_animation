import argparse
from genericpath import isdir
import os
import sys
import random
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import constants.constants as constants
from constants.constants_utils import read_params
from torch_dataset import TestSet
from utils.model_utils import find_model

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw
from dtaidistance import dtw


def get_path(epoch, output_path, evaluation_path):
    model_file = find_model(int(epoch)) 
    path_data_out = output_path + model_file[0:-3] + "/"
    if(not isdir(path_data_out)):
        raise Exception(path_data_out + "is not a directory")

    path_evaluation = evaluation_path + model_file[0:-3] + "/"
    if(not isdir(path_evaluation)):
        os.makedirs(path_evaluation, exist_ok=True)
    return model_file, path_data_out, path_evaluation

def getData(path_data_out, file_to_evaluate, all_features, features = None):
    test_set = TestSet()
    gened_seqs = []
    real_seqs = []
    for file in file_to_evaluate:
        pd_file = pd.read_csv(path_data_out+file+".csv")
        pd_file = pd_file[all_features]
        if features != None:
            pd_file = pd_file[features]
        gened_seqs.append(pd_file)

    if features != None:
        for test_video in test_set.Y_final_ori:
            real_seqs.append(test_video[features])
    else:
        real_seqs = test_set.Y_final_ori

    gened_frames = np.concatenate(gened_seqs, axis=0)
    real_frames = np.concatenate(real_seqs, axis=0)
    return real_frames, gened_frames

def getVideoData(file, index, all_features, features = None):
    test_set = TestSet()
    gened_seqs = []
    real_seqs = []
    pd_file = pd.read_csv(file)[all_features]

    if features != None:
        real_seqs.append(test_set.Y_final_ori[index][features])
        pd_file = pd_file[features]
    else:
        real_seqs = test_set.Y_final_ori[index]
    gened_seqs.append(pd_file)
    gened_frames = np.concatenate(gened_seqs, axis=0)
    real_frames = np.concatenate(real_seqs, axis=0)
    return real_frames, gened_frames

def check_dataset(datasets):
    print("selection of dataset : ", datasets)
    if(len(datasets) != 1):
        sys.exit("Only one dataset at a time for the generation")
    elif("trueness" in constants.datasets[0]):
        #check in create_set.py file
        order_in_y_final = ["scene2_confrontation_prise1_mic1",
        "scene2_confrontation_prise1_mic2",
        "scene2_sexisme_mic1",
        "scene2_sexisme_mic2",
        "scene3_sexisme_prise3_mic1",
        "scene3_sexisme_prise3_mic2",
        "scene5_confrontation1_mic1",
        "scene5_confrontation1_mic2"]
    elif("cheese" in constants.datasets[0]):
        order_in_y_final = ["AW-CG_mic1",
        "AW-CG_mic2",
        "AC-MZ_mic1",
        "AC-MZ_mic2",
        "ER-AG_mic1",
        "ER-AG_mic2"]
    else:
        sys.exit("Only cheese and trueness for the dataset evaluation are ready")
    
    return order_in_y_final

def compute_jerks(data, dim=3):
    # return jerks of each joint averaged over all frames
    # Third derivative of position is jerk
    jerks = np.diff(data, n=3, axis=0)

    num_jerks = jerks.shape[0]
    num_joints = jerks.shape[1] // dim
    jerk_norms = np.zeros((num_jerks, num_joints))

    for i in range(num_jerks):
        for j in range(num_joints):
            x1 = j * dim + 0
            x2 = j * dim + dim
            jerk_norms[i, j] = np.linalg.norm(jerks[i, x1:x2])
    average = np.mean(jerk_norms, axis=0)

    # Take into account that frame rate was 25 fps
    scaled_av = average * 25 * 25 * 25

    return scaled_av

def compute_acceleration(data, dim=3):
    # Second derivative of position is acceleration
    accs = np.diff(data, n=2, axis=0)

    num_accs = accs.shape[0]
    num_joints = accs.shape[1] // dim

    acc_norms = np.zeros((num_accs, num_joints))

    for i in range(num_accs):
        for j in range(num_joints):
            x1 = j * dim + 0
            x2 = j * dim + dim
            acc_norms[i, j] = np.linalg.norm(accs[i, x1:x2])

    average = np.mean(acc_norms, axis=0)

    # Take into account that frame rate was 25 fps
    scaled_av = average * 25 * 25

    return scaled_av


def create_pca(real_frames, gened_frames, pdf, features_name = ""):
    #first scaling the data
    scaler = StandardScaler()
    scale_real = scaler.fit(real_frames)
    X_real = scale_real.transform(real_frames)
    X_gened = scale_real.transform(gened_frames)


    mypca = PCA(n_components=2, random_state=1) # calculate the two major components

    #the covariance matrix
    x_real_covariance_matrix = np.cov(np.transpose(np.array(X_real)))
    x_gened_covariance_matrix = np.cov(np.transpose(np.array(X_gened)))

    data_real = mypca.fit(x_real_covariance_matrix).transform(x_real_covariance_matrix)
    data_generated = mypca.fit(x_gened_covariance_matrix).transform(x_gened_covariance_matrix)

    #to get the similarities, we concatenate the two vectors (verticaly) to get a unique vector for each dataset --> a unique vector for real and a unique vector for gened
    #then we calculate the pearson similarities 

    print('Valeur de variance', mypca.singular_values_, 'Explained variation per principal component: {}'.format(mypca.explained_variance_ratio_))

    df_real = pd.DataFrame(data = data_real, columns = ['principal component 1', 'principal component 2'])
    df_real_vert = pd.concat([df_real['principal component 1'], df_real['principal component 2']]).reset_index(drop=True)

    df_generated = pd.DataFrame(data = data_generated, columns = ['principal component 1', 'principal component 2'])
    df_generated_vert = pd.concat([df_generated['principal component 1'], df_generated['principal component 2']]).reset_index(drop=True)

    #similarity between the two dataset 
    print("correlation of data is", np.corrcoef(df_real_vert, df_generated_vert)[0,1])

    #pca in graphs
    pca_real = mypca.fit(X_real)
    data_real = pca_real.transform(X_real)
    data_generated = pca_real.transform(X_gened)
    df_real = pd.DataFrame(data = data_real, columns = ['principal component 1', 'principal component 2'])
    df_generated = pd.DataFrame(data = data_generated, columns = ['principal component 1', 'principal component 2'])
    indicesToKeep = df_generated.index

    plt.figure(figsize=(3, 3), dpi=100)
    plt.title('pca_'+features_name)
    plt.scatter(df_real.loc[indicesToKeep, 'principal component 1'], df_real.loc[indicesToKeep, 'principal component 2'], label='Real data', rasterized=True)
    plt.scatter(df_generated.loc[indicesToKeep, 'principal component 1'], df_generated.loc[indicesToKeep, 'principal component 2'], label='Generated data', alpha=0.7, rasterized=True)
    plt.xlabel('Principal Component - 1')
    plt.ylabel('Principal Component - 2')
    plt.legend()
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
    

def plot_figure(real_signal, generated_signal, pdf, features_name):
        x_real = range(len(real_signal))
        x_gen = range(len(generated_signal))
        plt.figure(figsize=(3, 3), dpi=100)
        plt.title(features_name)
        plt.plot(x_gen, generated_signal, label="generated", alpha=0.5, rasterized=True)
        plt.plot(x_real, real_signal, label="real", alpha=0.8, rasterized=True)
        plt.legend()
        pdf.savefig()
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-params', help='Path to the constant file', default="./params.cfg")
    parser.add_argument('-epoch', help='number of epoch of recovred model', default=9)
    parser.add_argument('-dataset', help='wich video we want to generate', default="")
    parser.add_argument('-dtw', action='store_true')
    parser.add_argument('-pca', action='store_true')
    parser.add_argument('-curve', action='store_true')
    parser.add_argument('-curveVideo', action='store_true')
    parser.add_argument('-acceleration', action='store_true')
    parser.add_argument('-jerk', action='store_true')

    args = parser.parse_args()
    read_params(args.params, "eval")

    if(args.dataset == ""):
        sys.exit("You have to indicate a dataset for the evaluation (only cheese or trueness and one at a time for the moment)")
    else:
        constants.datasets = args.dataset.split(",")

    file_to_evaluate = check_dataset(constants.datasets)
    model_file, path_data_out, path_evaluation = get_path(args.epoch, constants.output_path, constants.evaluation_path)


    all_features = ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y", "pose_Rx", "pose_Ry",
                "pose_Rz", "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]
    
    dim_features = ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "pose_Rx", "pose_Ry", "pose_Rz"]

    #creation of different mesures for each type of output
    types_output = {
    "first_eye" : ["gaze_0_x", "gaze_0_y", "gaze_0_z"],
    "second_eye" : ["gaze_1_x", "gaze_1_y", "gaze_1_z"],
    "gaze_angle" : ["gaze_angle_x", "gaze_angle_y"],
    "pose" : ["pose_Rx", "pose_Ry", "pose_Rz"],
    "sourcils" : ["AU01_r", "AU02_r", "AU04_r"],
    "visage" : ["AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r"],
    "bouche" : ["AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r"],
    "clignement" : ["AU45_r"]}
    
    if(args.acceleration):
        print("*"*10, "ACCELERATION", "*"*10)
        real_frames, gened_frames = getData(path_data_out, file_to_evaluate, all_features, dim_features)
        acc_gened = compute_acceleration(gened_frames)
        acc_real = compute_acceleration(real_frames)
        print("acceleration : real", acc_real, "gen ", acc_gened)

    if(args.jerk):
        print("*"*10, "JERK", "*"*10)
        real_frames, gened_frames = getData(path_data_out, file_to_evaluate, all_features, dim_features)
        jerk_gened = compute_jerks(gened_frames)
        jerk_real = compute_jerks(real_frames)
        print("jerks : real", jerk_real, "gen ", jerk_gened)

    if(args.curve):
        print("*"*10, "GENERAL CURVE", "*"*10)
        with PdfPages(path_evaluation + "curve.pdf") as pdf:
            for feature in all_features : 
                print("*"*5,feature, "*"*5)
                real_frames, gened_frames = getData(path_data_out, file_to_evaluate, all_features, feature)
                plot = plot_figure(real_frames, gened_frames, pdf, feature)


    
    if(args.curveVideo):
        print("*"*10,"VIDEO CURVE", "*"*10)
        for index, file in enumerate(file_to_evaluate):
            print("*"*5, file, "*"*5)
            with PdfPages(path_evaluation + file + "_curve.pdf") as pdf:
                for feature in all_features :
                    print("*"*2,feature, "*"*2)
                    real_frames, gened_frames = getVideoData(path_data_out+file+".csv", index, all_features, feature)
                    plot = plot_figure(real_frames, gened_frames, pdf, feature)


    if(args.dtw):
        print("*"*10,"DTW", "*"*10)
        mean_dist_tab = None #for all the video
        for index, file in enumerate(file_to_evaluate):
            print("*"*5, file, "*"*5)
            df_dtw = pd.DataFrame()  
            dist_tab = []
            for feature in all_features :
                print("*"*2,feature, "*"*2)
                real_frames, gened_frames = getVideoData(path_data_out+file+".csv", index, all_features, feature)
                real_frames, gened_frames = np.squeeze(real_frames), np.squeeze(gened_frames)
                distance = dtw.distance_fast(real_frames, gened_frames, use_pruning=True)
                dist_tab.append(distance)
            df_dtw = pd.DataFrame(dist_tab, index=all_features)
            df_dtw.to_csv(path_evaluation + file + "_dtw.csv")

            if(mean_dist_tab != None):
                mean_dist_tab = [mean_dist_tab[i] +  dist_tab[i] for i in range(len(dist_tab))]
            else:
                mean_dist_tab = dist_tab
            print(mean_dist_tab)

        print("*"*10,"GLOBAL DTW", "*"*10)
        mean_dist_tab = [x / len(file_to_evaluate) for x in mean_dist_tab]
        df_global_dtw = pd.DataFrame(mean_dist_tab, index=all_features)
        df_global_dtw.to_csv(path_evaluation + "global_dtw.csv", sep=";")


    if(args.pca):
        print("*"*10,"PCA", "*"*10)
        with PdfPages(path_evaluation + "PCA.pdf") as pdf:
            for cle, features in types_output.items():
                real_frames, gened_frames = getData(path_data_out, file_to_evaluate, all_features, features)
                print("*"*5,cle, "*"*5)
                if(cle != "clignement" and args.pca):
                    pca = create_pca(real_frames, gened_frames, pdf, cle)
    
