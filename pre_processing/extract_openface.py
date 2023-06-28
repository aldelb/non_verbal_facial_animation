import os
from os import listdir
from os.path import isfile, join
import sys
import subprocess
import pandas as pd
from scipy import ndimage
import numpy as np
import os
from pandas import DataFrame, concat

def getPath(dataset_name):
    set = None
    if dataset_name == "pom":
        video_dir = "Full/"
    elif dataset_name == "mosi":
        video_dir = "Full/"
    elif dataset_name == "mosei":
        video_dir = "Full/"
    elif dataset_name == "trueness":
        video_dir = "Full/"
        set = ["train", "test"]
    elif dataset_name == "obama":
         video_dir = "Full/"
         set = ["train", "test"]
    elif dataset_name == "cheese":
         video_dir = "Full/"
         set = ["train", "test"]
    else:
        sys.exit("Error in the dataset name")

    path = "/storage/raid1/homedirs/alice.delbosc/data/"+dataset_name+"_data/raw_data/Video/"
    processed_dir = "processed/"

    return path, processed_dir, video_dir, set


def createCsvFile(dir, out, openFace_dir):
    print("*"*10, "createCsvFile", "*"*10)
    to_create = []
    for f in listdir(dir):
        print(f)
        csv_file = f[0:-4] + ".csv"
        if not isfile(join(out, csv_file)):
            print(csv_file)
            to_create.append(csv_file)
        if not isfile(join(out, csv_file)):
            subprocess.Popen(openFace_dir + "/build/bin/FeatureExtraction -f " + join(dir, f) + ' -pose -aus -gaze -out_dir '+ out, shell=True).wait()

def replaceOrDropOutlierValue(out):
    print("*"*10, "replaceOutlierValue", "*"*10)
    for csv_file in listdir(out + "origin/"):
        if(".csv" in csv_file):
            print(csv_file)
            #replace the value if superior to 1 or inferior to -1
            df = pd.read_csv(join(out+ "origin/",csv_file))
            df['pose_Rx'] = np.where(df['pose_Rx'] > 1, 1, df['pose_Rx'])
            df['pose_Rx'] = np.where(df['pose_Rx'] < -1, -1, df['pose_Rx'])

            df['pose_Ry'] = np.where(df['pose_Ry'] > 1, 1, df['pose_Ry'])
            df['pose_Ry'] = np.where(df['pose_Ry'] < -1, -1, df['pose_Ry'])

            df['pose_Rz'] = np.where(df['pose_Rz'] > 1, 1, df['pose_Rz'])
            df['pose_Rz'] = np.where(df['pose_Rz'] < -1, -1, df['pose_Rz'])
            #drop if the sum sup to 1.2, replacement of those in the next step
            outlier = df.where(abs(df['pose_Rx']) + abs(df['pose_Ry']) + abs(df['pose_Rz']) > 1.2).dropna()
            lst = [*outlier.index] 
            outlier_index = list(dict.fromkeys(lst))
            df = df.drop(outlier_index)
            df.to_csv(join(out,csv_file), index=False)

def smoothCenter(out, visual_features):
    print("*"*10, "smoothCenter", "*"*10)
    for csv_file in listdir(out):
        if(".csv" in csv_file):
            print(csv_file)
            df = pd.read_csv(join(out,csv_file))
            df_smooth = df.copy()
            for column in ["pose_Rx", "pose_Ry", "pose_Rz", "gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y"]:
                df_smooth[column] = df_smooth[column] - df_smooth[column].mean()
            for column in visual_features:
                df_smooth[column] = ndimage.median_filter(df_smooth[column], size=7, mode="constant", cval=0)        
            df_smooth.to_csv(join(out,csv_file), index=False)

#calculate transition betwwen two timestep
def calculateMissingTransition(df, index, timestep):
    new_df = df.copy()
    number_of_missing = int(round(new_df.at[index+1, "timestamp"] - new_df.at[index, "timestamp"], 2)/timestep)
    step = (new_df.iloc[index+1] - new_df.iloc[index]) / number_of_missing
    old = new_df.iloc[index]
    for idx in range(index+1, index+number_of_missing):
        new_line = [old + step]
        line = DataFrame(new_line, columns = df.columns, index=[idx])
        new_df = concat([new_df.iloc[:idx], line, new_df.iloc[idx:]]).reset_index(drop=True)
        old = old + step
    return new_df, number_of_missing


def correctTimeAndTransitions(out, timestep, min_confidence):
    print("*"*10, "correctTimeAndTransitions", "*"*10)
    for csv_file in listdir(out):
        if(".csv" in csv_file):
            print(csv_file)
            df_video = pd.read_csv(join(out,csv_file))
            # we group by timestep (some are duplicated) and we create the missing timestep by calculating the transitions  
            df_video = df_video.groupby(by=["timestamp"]).mean()
            df_video = df_video.reset_index()

            # if confidence openFace < min_confidence 
            # --> drop the line
            old_len = len(df_video)
            max_index = len(df_video) - 1
            index = 1
            while(index < max_index):
                if df_video.at[index, "confidence"]  < min_confidence:
                    print("yes")
                    df_video.drop(index)
                index = index + 1
            df_video = df_video.reset_index(drop=True)
            print("confidence", old_len, len(df_video))
            


            # calculate missing timestep
            max_index = len(df_video) - 1
            index = 1
            index_to_modify = []
            while(index <= max_index):
                if round(df_video.at[index, "timestamp"],2)  != round(df_video.at[index-1, "timestamp"] + timestep ,2):
                    index_to_modify.append(index-1)
                index = index + 1

            total_add = 0
            old_len = len(df_video)
            for index in index_to_modify:
                df_video, add_line = calculateMissingTransition(df_video, index + total_add, timestep)
                total_add = total_add + add_line - 1
            print("transition", old_len, len(df_video))
            
            df_video.to_csv(join(out,csv_file), index=False)




if __name__ == "__main__":
    
    openFace_dir = "/storage/raid1/homedirs/alice.delbosc/Projects/Code/OpenFace/"

    dataset_name = sys.argv[1]
    with_extract = sys.argv[2] ## if False, reuse the existing csv file for replaceOutlierValue, correctTimeAndTransitions, smoothCenter
    
    visual_features  = ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y", "pose_Tx",	"pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry",
                    "pose_Rz", "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]
    timestep = 0.04
    min_confidence = 0.6
    path, processed_dir, video_dir, set = getPath(dataset_name)


    if(set == None):
        dir = path + video_dir
        out = path + processed_dir
        if(not os.path.exists(out)):
             os.mkdir(out)
        if(with_extract == "True"):
            createCsvFile(dir, out + "origin/", openFace_dir)
        replaceOrDropOutlierValue(out)
        correctTimeAndTransitions(out, timestep, min_confidence)
        smoothCenter(out, visual_features)
    
    else:
         for set_name in set:
            dir = join(path, video_dir, set_name)
            out = join(path, processed_dir, set_name)
            if(not os.path.exists(out)):
                os.mkdir(out)
            if(with_extract == "True"):
                createCsvFile(dir, out + "origin/", openFace_dir)
            replaceOrDropOutlierValue(out)
            correctTimeAndTransitions(out, timestep, min_confidence)
            smoothCenter(out, visual_features)