import argparse
import os
import sys
import numpy as np
import pandas as pd
import numpy as np
import pickle
from genericpath import isfile

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

def create_set(dataset_name, output_path, visual_path, audio_path, collection, type_of_set, silence_path, audioToZero, moveToZero):
    #toutes les 4 sec, avec recouvrement 10 frames
    segment_len = 100
    overlap = 10
    X = []
    Y = []
    intervals = []
    row_silence = pd.read_csv(silence_path)[audio_features[1:]].iloc[0].values
    row_silence = np.insert(row_silence, 0, 0)
    for key in collection:
        path_df_audio = os.path.join(audio_path, key + ".csv")
        path_df_visual = os.path.join(visual_path, key + ".csv")
        if(os.path.isfile(path_df_audio) and os.path.isfile(path_df_visual)):
            print("process of", key)
            df_audio = pd.read_csv(path_df_audio)[audio_features]
            df = pd.read_csv(path_df_visual)
            df_video = df[visual_features].copy()
            df_timestamp = df[["timestamp"]].copy()

            index_not_move = df_audio.where(df_audio.iloc[:,0] == 0).dropna().index

            if audioToZero:
                df_audio.iloc[index_not_move] = row_silence #for audio we put to silence if not speaking
            
            if moveToZero:
                index_not_move = [i for i in index_not_move if i <= len(df_video)-1] #for video we put all move to 0 if not speaking
                df_video.iloc[index_not_move] = 0

            i_max = len(df_video)
            i = 0
            while(i+segment_len < i_max):
                X.append(df_audio.loc[i:i+segment_len-1,:].values) 
                Y.append(df_video.loc[i:i+segment_len-1,:].values) 
                
                arr1 = df_timestamp.loc[i:i+segment_len-1]
                arr2 = arr1 + 0.04
                interval_tab = np.stack((arr1, arr2), axis=1)
                intervals.append([key, interval_tab])
                
                i += segment_len - overlap

    with open(output_path + "X_"+type_of_set+"_"+dataset_name+".p", 'wb') as f:
        pickle.dump(X, f)
    with open(output_path + "y_"+type_of_set+"_"+dataset_name+".p", 'wb') as f:
        pickle.dump(Y, f)
    with open(output_path + "intervals_"+type_of_set+"_"+dataset_name+".p", 'wb') as f:
        pickle.dump(intervals, f)


def create_final_y_test(test_key, processed_video_path, output_path, dataset_name):
        # we create one file per video 
        output_path = output_path + "y_test_final_"+dataset_name+".p"
        all_df = []
        for file in os.listdir(processed_video_path):
            if(file[-4:] == ".csv" and file[0:-4] in test_key):
                df = pd.read_csv(processed_video_path + file)[visual_features]
                all_df.append(df)
        with open(output_path, 'wb') as f:
            pickle.dump(all_df, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset')
    parser.add_argument('-silenceAudio', action='store_true')
    parser.add_argument('-zeroMove', action='store_true')
    args = parser.parse_args()

    dataset_name = args.dataset
    audioToZero = args.silenceAudio
    moveToZero = args.zeroMove

    final_name = dataset_name
    if(audioToZero and moveToZero):
        final_name = final_name + "_moveAndAudioSpeakerOnly"
    elif(audioToZero):
        final_name = final_name + "_audioSpeakerOnly"
    elif(moveToZero):
        final_name = final_name + "_moveSpeakerOnly"
    
    print("final name :", final_name)
    
    general_path = "/storage/raid1/homedirs/alice.delbosc/"
    dataset_path = general_path+ "data/" +dataset_name+"_data/"
    keys_path = dataset_path + "raw_data/Video/Full/"
    output_path = general_path + "Projects/Data/without_context/"
    silence_path = general_path + "Projects/Code/without_context/non-verbal-behaviours-generation/pre_processing/silence/silence.csv"


    sets = ["train", "test"]


    for set_name in sets:
        print("management of ", set_name)
        keys = os.listdir(keys_path+set_name)
        keys = [x.split('.')[0] for x in keys]

        audio_path = dataset_path + "raw_data/Audio/processed/" + set_name + "/"
        visual_path = dataset_path + "raw_data/Video/processed/" + set_name + "/"

        create_set(final_name, output_path, visual_path, audio_path, keys, set_name, silence_path, audioToZero, moveToZero)
        print("*"*10, set_name, " set created", "*"*10)

        if(set_name == "test"):
            create_final_y_test(keys, visual_path, output_path, final_name)
            print("*"*10, "final test file created", "*"*10)
            print("*"*10, "end", "*"*10)

    return 0

if __name__ == "__main__":
    sys.exit(main())
