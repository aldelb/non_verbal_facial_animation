import os
from os import listdir
from os.path import isfile, join
import sys
import subprocess
import opensmile
import speechpy
import numpy as np
import pandas as pd
from scipy import ndimage
import pickle

audio_features = ["Speak", "Loudness_sma3","alphaRatio_sma3","hammarbergIndex_sma3","slope0-500_sma3","slope500-1500_sma3","spectralFlux_sma3","mfcc1_sma3","mfcc2_sma3","mfcc3_sma3","mfcc4_sma3",\
                  "F0semitoneFrom27.5Hz_sma3nz","jitterLocal_sma3nz","shimmerLocaldB_sma3nz","HNRdBACF_sma3nz","logRelF0-H1-H2_sma3nz"\
                  ,"logRelF0-H1-A3_sma3nz","F1frequency_sma3nz","F1bandwidth_sma3nz","F1amplitudeLogRelF0_sma3nz","F2frequency_sma3nz","F2bandwidth_sma3nz",\
                    "F2amplitudeLogRelF0_sma3nz","F3frequency_sma3nz","F3bandwidth_sma3nz","F3amplitudeLogRelF0_sma3nz","first_Loudness_sma3","second_Loudness_sma3",\
                        "first_alphaRatio_sma3","second_alphaRatio_sma3","first_hammarbergIndex_sma3","second_hammarbergIndex_sma3","first_slope0-500_sma3",\
                            "second_slope0-500_sma3","first_slope500-1500_sma3","second_slope500-1500_sma3","first_spectralFlux_sma3","second_spectralFlux_sma3",\
                                "first_mfcc1_sma3","second_mfcc1_sma3","first_mfcc2_sma3","second_mfcc2_sma3","first_mfcc3_sma3","second_mfcc3_sma3","first_mfcc4_sma3"\
                                    ,"second_mfcc4_sma3","first_F0semitoneFrom27.5Hz_sma3nz","second_F0semitoneFrom27.5Hz_sma3nz","first_jitterLocal_sma3nz","second_jitterLocal_sma3nz",\
                                        "first_shimmerLocaldB_sma3nz","second_shimmerLocaldB_sma3nz","first_HNRdBACF_sma3nz","second_HNRdBACF_sma3nz","first_logRelF0-H1-H2_sma3nz",\
                                            "second_logRelF0-H1-H2_sma3nz","first_logRelF0-H1-A3_sma3nz","second_logRelF0-H1-A3_sma3nz","first_F1frequency_sma3nz","second_F1frequency_sma3nz",\
                                                "first_F1bandwidth_sma3nz","second_F1bandwidth_sma3nz","first_F1amplitudeLogRelF0_sma3nz","second_F1amplitudeLogRelF0_sma3nz","first_F2frequency_sma3nz",\
                                                    "second_F2frequency_sma3nz","first_F2bandwidth_sma3nz","second_F2bandwidth_sma3nz","first_F2amplitudeLogRelF0_sma3nz",\
                                                        "second_F2amplitudeLogRelF0_sma3nz","first_F3frequency_sma3nz","second_F3frequency_sma3nz","first_F3bandwidth_sma3nz",\
                                                            "second_F3bandwidth_sma3nz","first_F3amplitudeLogRelF0_sma3nz","second_F3amplitudeLogRelF0_sma3nz"]

visual_features = ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y", "pose_Rx", "pose_Ry",
                "pose_Rz", "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]



def getPath():
    path = "/gpfsdswork/projects/rech/urk/uln35en/Data/audio_file/"
    raw_dir = "raw_data/"
    processed_dir = "processed/"
    anno_dir = "annotation/"
    set_dir = "set/"

    return path, raw_dir, processed_dir, anno_dir, set_dir


def addDerevative(tab, audio_features):
        for audio in audio_features:
                first = speechpy.processing.derivative_extraction(np.array(tab[[audio]]), 1)
                second = speechpy.processing.derivative_extraction(first, 1)
                tab["first_"+audio] = first
                tab["second_"+audio] = second
        return tab

def createCsvFile(wav_file, origin_processed_file):
    print("*"*10, "createCsvFile", "*"*10)
    to_create = []
    if not isfile(origin_processed_file):
        smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                )
        result_smile = smile.process_file(wav_file)
        result_smile.to_csv(origin_processed_file, sep=',')

def addDerivative(processed_file, audio_features):
    print("*"*10, "smoothAndAddDerivative", "*"*10)
    df = pd.read_csv(processed_file)
    df = addDerevative(df, audio_features)
    df.to_csv(processed_file, index=False)

def removeOverlapAndAlignStep(origin_processed_file, processed_file, objective_step, audio_features):
    print("*"*10, "removeOverlapAndAlignStep", "*"*10)
    df_audio = pd.read_csv(origin_processed_file)
    df_audio["start"] = df_audio["start"].transform(lambda x :  pd.Timedelta(x).total_seconds())
    df_audio["end"] = df_audio["end"].transform(lambda x :  pd.Timedelta(x).total_seconds())
    #we remove the overlaps and recalculate with averages
    df_audio_wt_overlap = df_audio.copy()
    df_audio_wt_overlap = df_audio_wt_overlap.rename(columns={"start":"timestamp"})
    df_audio_wt_overlap = df_audio_wt_overlap.drop(columns=["end"])
    df_audio_wt_overlap = df_audio_wt_overlap[audio_features]
    #mean the value of overlap
    for index, row in df_audio_wt_overlap.iterrows():
        if index != 0:
            df_audio_wt_overlap.at[index, "Loudness_sma3"] = (row["Loudness_sma3"] + df_audio.iloc[[index-1]]["Loudness_sma3"]) / 2

    #we change the timestep to match the openface timestep
    df_audio_wt_overlap["timestamp"] = df_audio_wt_overlap["timestamp"].astype(float)
    df_audio_wt_overlap["timestamp"] = ((df_audio_wt_overlap["timestamp"]/objective_step).astype(int)) * objective_step
    df_audio_wt_overlap["timestamp"] = round(df_audio_wt_overlap["timestamp"],2)
    df_audio_wt_overlap = df_audio_wt_overlap.groupby(by=["timestamp"]).mean()
    df_audio_wt_overlap = df_audio_wt_overlap.reset_index()
    df_audio = df_audio_wt_overlap
    df_audio.to_csv(processed_file, index=False)
    
def addSpeakOrNotFeatures(processed_file, anno_file, timestep):
    print("*"*10, "addSpeakOrNotFeatures", "*"*10)
    df_audio = pd.read_csv(processed_file)
    df_anno = pd.read_csv(anno_file,  names=["features", "begin", "end", "speak"])
    df_anno = df_anno[df_anno['features'] == "IPUs"].reset_index(drop = True)
    df_anno['speak'] = np.where(df_anno['speak']!= '#', 1, 0)

    #we add the speaking or not speaking features to the audio features
    df_anno = df_anno[["begin", "end", "speak"]]

    #we change the timestep to match the openface timestep
    objective_step = timestep
    current_time = 0
    last_index = len(df_anno)-1
    new_anno = pd.DataFrame([], columns=["timestamp", "speak"])
    index = 0
    while(index <= last_index):
        while(current_time + objective_step <= df_anno.at[index, "end"]):
            new_row = pd.Series({'timestamp': current_time, 'speak': df_anno.at[index, "speak"]})
            new_anno = pd.concat([new_anno, new_row.to_frame().T], ignore_index=True)
            current_time = round(current_time + objective_step,2)
        index = index + 1
        
        #add the speaking or not column to audio dataframe
        df_audio.insert(1, "Speak", new_anno["speak"], True)
        df_audio.to_csv(processed_file, index=False)


def create_set(filename, processed_file, set_file, set_interval_file):
    print("*"*10, "create_set", "*"*10)
    #toutes les 4 sec, avec recouvrement 10 frames
    segment_len = 100
    overlap = 10
    X = []
    intervals = []

    path_df_audio = processed_file
    if(os.path.isfile(path_df_audio)):
        print("process of", path_df_audio)
        df = pd.read_csv(path_df_audio)
        df_timestamp = df[["timestamp"]].copy()
        print(df)
        df_audio = df[audio_features].copy()
        
        i_max = len(df_audio)
        i = 0
        while(i+segment_len < i_max):
            X.append(df_audio.loc[i:i+segment_len-1,:].values) 
            
            arr1 = df_timestamp.loc[i:i+segment_len-1]
            arr2 = arr1 + 0.04
            interval_tab = np.stack((arr1, arr2), axis=1)
            intervals.append([filename, interval_tab])
            
            i += segment_len - overlap

    with open(set_file, 'wb') as f:
        pickle.dump(X, f)
    with open(set_interval_file, 'wb') as f:
        pickle.dump(intervals, f)



if __name__ == "__main__":
    with_extract = sys.argv[1] ## if False, reuse the existing csv file for removeOverlapAndAlignStep and addDerivative
    file_name = sys.argv[2]

    small_audio_features = ["timestamp", 'Loudness_sma3','alphaRatio_sma3','hammarbergIndex_sma3','slope0-500_sma3','slope500-1500_sma3','spectralFlux_sma3','mfcc1_sma3',\
                      'mfcc2_sma3','mfcc3_sma3','mfcc4_sma3','F0semitoneFrom27.5Hz_sma3nz','jitterLocal_sma3nz','shimmerLocaldB_sma3nz','HNRdBACF_sma3nz',\
                        'logRelF0-H1-H2_sma3nz','logRelF0-H1-A3_sma3nz','F1frequency_sma3nz','F1bandwidth_sma3nz','F1amplitudeLogRelF0_sma3nz','F2frequency_sma3nz',\
                            'F2bandwidth_sma3nz','F2amplitudeLogRelF0_sma3nz','F3frequency_sma3nz','F3bandwidth_sma3nz','F3amplitudeLogRelF0_sma3nz']
    
    timestep = 0.04
    path, raw_dir, processed_dir, anno_dir, set_dir = getPath()

    wav_file = join(path, raw_dir, file_name+".wav")
    anno_file = join(path, anno_dir, file_name+".csv")
    processed_file = join(path, processed_dir, file_name+".csv")
    origin_processed_file = join(path, processed_dir, "origin", file_name+".csv")
    set_file = join(path, set_dir, file_name + ".p")
    set_interval_file = join(path, set_dir, file_name+"_interval.p")

    if(with_extract == "True"):
        createCsvFile(wav_file, origin_processed_file)
    removeOverlapAndAlignStep(origin_processed_file, processed_file, timestep, small_audio_features)
    addSpeakOrNotFeatures(processed_file, anno_file, timestep)
    addDerivative(processed_file, small_audio_features[1:])

    create_set(file_name, processed_file, set_file, set_interval_file)
