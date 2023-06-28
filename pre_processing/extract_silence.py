import os
from os import listdir
from os.path import isfile, join
import sys
import subprocess
import opensmile
import speechpy
import numpy as np

def addDerevative(tab, audio_features):
        for audio in audio_features:
                first = speechpy.processing.derivative_extraction(np.array(tab[[audio]]), 1)
                second = speechpy.processing.derivative_extraction(first, 1)
                tab["first_"+audio] = first
                tab["second_"+audio] = second
        return tab


if __name__ == "__main__":
        silence_dir = "./silence/"
        audio_dir = silence_dir + "silence.wav"
        audio_features = ['Loudness_sma3','alphaRatio_sma3','hammarbergIndex_sma3','slope0-500_sma3','slope500-1500_sma3','spectralFlux_sma3','mfcc1_sma3','mfcc2_sma3','mfcc3_sma3','mfcc4_sma3','F0semitoneFrom27.5Hz_sma3nz','jitterLocal_sma3nz','shimmerLocaldB_sma3nz','HNRdBACF_sma3nz','logRelF0-H1-H2_sma3nz','logRelF0-H1-A3_sma3nz','F1frequency_sma3nz','F1bandwidth_sma3nz','F1amplitudeLogRelF0_sma3nz','F2frequency_sma3nz','F2bandwidth_sma3nz','F2amplitudeLogRelF0_sma3nz','F3frequency_sma3nz','F3bandwidth_sma3nz','F3amplitudeLogRelF0_sma3nz']
        smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
        result_smile = smile.process_file(audio_dir)
        result_smile = addDerevative(result_smile, audio_features)
        result_smile.to_csv(join(silence_dir, "silence.csv"), sep=',')
