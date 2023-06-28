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

def getPath(dataset_name):
    set = None
    anno_dir = None
    cut_dictionnary = None
    if dataset_name == "pom":
        audio_dir = "Full"
    elif dataset_name == "mosi":
        audio_dir = "WAV_16000/Full"
    elif dataset_name == "mosei":
        audio_dir = "Full/WAV_16000"
    elif dataset_name == "trueness":
        audio_dir = "WAV_16000/Full"
        anno_dir = "Annotations/"
        set = ["train", "test"]
        cut_dictionnary={'scene1_sexisme' : [9,176], 'scene2_sexisme' : [18,180],'scene2_confrontation_prise1' : [6,414],'scene2_confrontation2' : [5,420], 'scene2_confrontation3_prise3' : [8,303], 
        'scene2_confrontation4' : [5,433], 'scene2_confrontation5_prise2' : [5,418], 'scene3_sexisme_prise3' : [5,219], 'scene3_confrontation1' : [5,455], 'scene3_confrontation2' : [6,428], 
        'scene3_confrontation3' : [5,433], 'scene3_confrontation5_prise2' : [5,358], 'scene4_racisme_prise2' : [5,231], 'scene5_confrontation1' : [4,327], 'scene6_confrontation2' : [4,436],  
        'scene7_confrontation3' : [6,537], 'scene8_confrontation4' : [5,477],  'scene9_confrontation5' : [5,439]}

    elif dataset_name == "obama":
         audio_dir = "Full"
         set = ["train", "test"]

    elif dataset_name == "cheese":
        audio_dir = "WAV_16000/Full"
        anno_dir = "Annotations/"
        set = ["train", "test"]
        cut_dictionnary={'AA-OR' : [86,960], 'AC-MZ' : [120,890],'AW-CG' : [132,1020],'CM-MCC' : [30,960], 'ER-AG' : [120,880], 
        'FB-CB' : [90,1100], 'JS-CL' : [110,975], 'LP-MA' : [347,1173], 'MA-PC' : [150,1035], 'MD-AD' : [180,1151], 
        'PR-EM' : [120,1104]}
    
    else:
        sys.exit("Error in the dataset name")

    path = "/storage/raid1/homedirs/alice.delbosc/data/"+dataset_name+"_data/raw_data/Audio/"
    processed_dir = "processed/"

    return path, processed_dir, audio_dir, anno_dir, set, cut_dictionnary


def addDerevative(tab, audio_features):
        for audio in audio_features:
                first = speechpy.processing.derivative_extraction(np.array(tab[[audio]]), 1)
                second = speechpy.processing.derivative_extraction(first, 1)
                tab["first_"+audio] = first
                tab["second_"+audio] = second
        return tab

def createCsvFile(dir, out):
    print("*"*10, "createCsvFile", "*"*10)
    to_create = []
    for f in listdir(dir):
        if(".wav" in f):
            csv_file = f[0:-4] + ".csv"
            if not isfile(join(out, csv_file)):
                print(join(dir,f))
                to_create.append(csv_file)
                smile = opensmile.Smile(
                        feature_set=opensmile.FeatureSet.eGeMAPSv02,
                        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                        )
                result_smile = smile.process_file(join(dir,f))
                result_smile.to_csv(join(out, csv_file), sep=',')

def addDerivative(out, audio_features):
    print("*"*10, "smoothAndAddDerivative", "*"*10)
    #df_silence = pd.read_csv("silence/silence.csv")
    for csv_file in listdir(out):
        print(csv_file)
        df = pd.read_csv(join(out,csv_file))
        # df_smooth = df.copy()
        # for column in audio_features:
        #     df_smooth[column] = ndimage.median_filter(df_smooth[column], size=7, mode="constant", cval=df_silence[column][0])
        df = addDerevative(df, audio_features)
        df.to_csv(join(out,csv_file), index=False)

def removeOverlapAndAlignStep(out, objective_step):
    print("*"*10, "removeOverlapAndAlignStep", "*"*10)
    for csv_file in listdir(out + "origin/"):
        print(csv_file)
        df_audio = pd.read_csv(join(out + "origin/",csv_file))
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
        df_audio.to_csv(join(out,csv_file), index=False)
    
def addSpeakOrNotFeatures(out, path, anno_dir, timestep, cut_dictionnary, dataset_name):
    print("*"*10, "addSpeakOrNotFeatures", "*"*10)
    for csv_file in listdir(out):
        print(csv_file)
        df_audio = pd.read_csv(join(out,csv_file))
        df_anno = pd.read_csv(join(path, anno_dir, csv_file),  names=["features", "begin", "end", "speak"])
        df_anno = df_anno[df_anno['features'] == "Tokens"].reset_index(drop = True)
        df_anno['speak'] = np.where(df_anno['speak']!= '#', 1, 0)

        #we add the speaking or not speaking features to the audio features
        df_anno = df_anno[["begin", "end", "speak"]]

        if(cut_dictionnary != None):
            cut_index = csv_file.find("mic") - 1
            key_cut = csv_file[0:cut_index]
            begin_cut = cut_dictionnary[key_cut][0]
            end_cut = cut_dictionnary[key_cut][1]

            #we cut a the end 
            last_index = len(df_anno)-1
            current_idx = last_index
            new_anno = df_anno.copy()
            while(df_anno.at[current_idx, "end"] > end_cut and df_anno.at[current_idx, "begin"] > end_cut):
                new_anno = new_anno.drop(index = current_idx)
                current_idx = current_idx - 1
                
            new_anno.at[current_idx, "end"] = end_cut

            #we cut the beginning
            begin_index = 0
            current_idx = begin_index
            while(df_anno.at[current_idx, "begin"] < begin_cut and df_anno.at[current_idx, "end"] < begin_cut):
                new_anno = new_anno.drop(index = current_idx)
                current_idx = current_idx + 1
                
            new_anno.at[current_idx, "begin"] = begin_cut


            #we shift all the temporal value with the begin value
            new_anno["begin"] = new_anno["begin"] - begin_cut
            new_anno["end"] = new_anno["end"] - begin_cut
            
            df_anno = new_anno.reset_index()

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
        df_audio.to_csv(join(out,csv_file), index=False)


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    with_extract = sys.argv[2] ## if False, reuse the existing csv file for removeOverlapAndAlignStep and addDerivative

    audio_features = ["timestamp", 'Loudness_sma3','alphaRatio_sma3','hammarbergIndex_sma3','slope0-500_sma3','slope500-1500_sma3','spectralFlux_sma3','mfcc1_sma3',\
                      'mfcc2_sma3','mfcc3_sma3','mfcc4_sma3','F0semitoneFrom27.5Hz_sma3nz','jitterLocal_sma3nz','shimmerLocaldB_sma3nz','HNRdBACF_sma3nz',\
                        'logRelF0-H1-H2_sma3nz','logRelF0-H1-A3_sma3nz','F1frequency_sma3nz','F1bandwidth_sma3nz','F1amplitudeLogRelF0_sma3nz','F2frequency_sma3nz',\
                            'F2bandwidth_sma3nz','F2amplitudeLogRelF0_sma3nz','F3frequency_sma3nz','F3bandwidth_sma3nz','F3amplitudeLogRelF0_sma3nz']
    

    
    timestep = 0.04
    path, processed_dir, audio_dir, anno_dir, set, cut_dictionnary = getPath(dataset_name)

    if(set == None):
        dir = path + audio_dir
        out = path + processed_dir
        if(not os.path.exists(out)):
             os.mkdir(out)
        if(with_extract == "True"):
            createCsvFile(dir, out + "origin/")
        removeOverlapAndAlignStep(out, timestep)
        addDerivative(out, audio_features[1:])
    
    else:
         for set_name in set:
            dir = join(path, audio_dir, set_name)
            out = join(path, processed_dir, set_name)
            if(not os.path.exists(out)):
                os.mkdir(out)
            if(not os.path.exists(out + "origin/")):
                os.mkdir(out + "origin/")
            if(with_extract == "True"):
                createCsvFile(dir, out + "origin/")
            removeOverlapAndAlignStep(out, timestep)
            addSpeakOrNotFeatures(out, path, anno_dir, timestep, cut_dictionnary, dataset_name)
            addDerivative(out, audio_features[1:])