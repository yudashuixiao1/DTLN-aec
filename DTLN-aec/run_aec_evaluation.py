# -*- coding: utf-8 -*-
"""
用来测试DTLN_aec模型的效果

Script to process a folder of .wav files with a trained DTLN model. 
This script supports subfolders and names the processed files the same as the 
original. The model expects 16kHz audio .wav files. Files with other 
sampling rates will be resampled. Stereo files will be downmixed to mono. 立体声文件将被混合成单声道。

The idea of this script is to use it for baseline or comparison purpose.

Example call:
    $python run_evaluation.py -i /name/of/input/folder  \
                              -o /name/of/output/folder \
                              -m /name/of/the/model.h5

python run_aec_evaluation.py -i /data/projects/AEC-Challenge/mydatasets/aec_testset_v7/result2/  \
                             -o /data/projects/AEC-Challenge/mydatasets/aec_testset_v7/result2/ \
                             -m /home/ljl/projects/DTLN/DTLN-aec_v2/models_DTLN_aec_v2_s_512_norm/DTLN_aec_v2_s_512_norm.h5

python run_aec_evaluation.py -i /data/projects/AEC-Challenge/mydatasets/models_test/漏回声误消验证_2/ \
                             -o /data/projects/AEC-Challenge/mydatasets/models_test/漏回声误消验证_2/ \
                             -m /home/ljl/projects/DTLN/DTLN-aec_v2/models_DTLN_aec_v2_s_512_norm/DTLN_aec_v2_s_512_norm.h5

python run_aec_evaluation.py -i /data/projects/AEC-Challenge/mydatasets/models_test/漏回声误消验证/  \
                             -o /data/projects/AEC-Challenge/mydatasets/models_test/漏回声误消验证/ \
                             -m /home/ljl/projects/DTLN/DTLN-aec_v2/models_DTLN_aec_v2_s_512_norm/DTLN_aec_v2_s_512_norm.h5
                             -m /home/ljl/projects/DTLN/DTLN-aec_v2/models_DTLN_aec_v4_1000h_512_norm/DTLN_aec_v4_1000h_512_norm.h5
                             
                             
                             -m /home/ljl/projects/DTLN/DTLN-aec_v2/models_DTLN_aec_v4_512_norm/DTLN_aec_v4_512_norm.h5

python run_aec_evaluation.py -i /data/projects/AEC-Challenge/mydatasets/models_test/CP965测试数据/  \
                             -o /data/projects/AEC-Challenge/mydatasets/models_test/CP965测试数据/ \
                             -m /home/ljl/projects/DTLN/DTLN-aec_v2/models_DTLN_aec_v2_s_512_norm/DTLN_aec_v2_s_512_norm.h5
                             -m /home/ljl/projects/DTLN/DTLN-aec_v2/models_DTLN_aec_v4_1000h_512_norm/DTLN_aec_v4_1000h_512_norm.h5
                             
                             
                             -m /home/ljl/projects/DTLN/DTLN-aec_v2/models_DTLN_aec_v4_512_norm/DTLN_aec_v4_512_norm.h5
                              

python run_aec_evaluation.py -i /data/projects/AEC-Challenge/mydatasets/models_test/软端v4x/  \
                             -o /data/projects/AEC-Challenge/mydatasets/models_test/软端v4x/ \
                             -m /home/ljl/projects/DTLN/DTLN-aec_v2/models_DTLN_aec_v4_1000h_512_norm/DTLN_aec_v4_1000h_512_norm.h5
                             -m /home/ljl/projects/DTLN/DTLN-aec_v2/models_DTLN_aec_v4_s_512_no_norm/DTLN_aec_v4_s_512_no_norm.h5
                             -m /home/ljl/projects/DTLN/DTLN-aec_v2/models_DTLN_aec_v4_s_512_norm/DTLN_aec_v4_s_512_norm.h5
                             
                             -m /home/ljl/projects/DTLN/DTLN-aec_v2/models_DTLN_aec_v4_512_norm/DTLN_aec_v4_512_norm.h5
                              -m /home/ljl/projects/DTLN/DTLN-aec_v2/models_DTLN_aec_v2_s_512_norm/DTLN_aec_v2_s_512_norm.h5
                              -m /home/ljl/projects/DTLN/DTLN-aec_mod/models_DTLN_aec_model_128/DTLN_aec_model_128.h5
                              -m /home/ljl/projects/DTLN/DTLN-aec_mod/models_DTLN_aec_model_256/DTLN_aec_model_256.h5
                              -m /home/ljl/projects/DTLN/DTLN-aec_mod/models_DTLN_aec_model_512/DTLN_aec_model_512.h5
                              
python run_aec_evaluation.py -i /data/projects/AEC-Challenge/mydatasets/models_test/综合测试10组代表性数据_s/  \
                              -o /data/projects/AEC-Challenge/mydatasets/models_test/综合测试10组代表性数据_s/ \
                              -m /home/ljl/projects/DTLN/DTLN-aec_v2/models_DTLN_aec_v4_s_512_norm/DTLN_aec_v4_s_512_norm.h5
                              -m /home/ljl/projects/DTLN/DTLN-aec_v2/models_DTLN_aec_v4_1000h_512_norm/DTLN_aec_v4_1000h_512_norm.h5
                              -m /home/ljl/projects/DTLN/DTLN-aec_v2/models_DTLN_aec_v4_s_512_no_norm/DTLN_aec_v4_s_512_no_norm.h5
                              -m /home/ljl/projects/DTLN/DTLN-aec_v2/models_DTLN_aec_v4_512_norm/DTLN_aec_v4_512_norm.h5
                              -m /home/ljl/projects/DTLN/DTLN-aec_v2/models_DTLN_aec_v2_s_512_norm/DTLN_aec_v2_s_512_norm.h5

This code is licensed under the terms of the MIT-license.
"""

import soundfile as sf
import librosa
import numpy as np
import os
import argparse
from DTLN_model import DTLN_model

# use the GPU with idx 0
os.environ["CUDA_VISIBLE_DEVICES"]='2,3'
# activate this for some reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'

def process_file(model, audio_file_name, out_file_name):
    '''
    单个.wav文件的处理
    Funtion to read an audio file, rocess it by the network and write the 
    enhanced audio to .wav file.

    Parameters
    ----------
    model : Keras model
        Keras model, which accepts audio in the size (1,timesteps).
    audio_file_name : STRING
        Name and path of the input audio file.
    out_file_name : STRING
        Name and path of the target file.

    '''
    '''
    # read audio file with librosa to handle resampling and enforce mono
    mic, fs = librosa.core.load(audio_file_name, sr=16000, mono=True)
    lpb, fs_2 = librosa.core.load(audio_file_name.replace(
        "mic.wav", "lpb.wav"), sr=16000, mono=True)
    '''
    #sig, fs = librosa.core.load(audio_file_name, sr=16000, mono=False)
    #print("shape of sig:",np.shape(sig))
    sig, fs = sf.read(audio_file_name)
    mic = sig[:,3] #2:mic   0:线性处理后的
    lpb = sig[:,1]
    if len(mic.shape) > 1 or len(lpb.shape) > 1:
        raise ValueError("Only single channel files are allowed.")
        # check for unequal length
    if len(lpb) > len(mic):
        lpb = lpb[: len(mic)]
    if len(lpb) < len(mic):
        mic = mic[: len(lpb)]
    len_orig = max(len(mic), len(lpb))
    zero_pad = np.zeros(384)
    mic = np.concatenate((zero_pad, mic, zero_pad), axis=0)
    lpb = np.concatenate((zero_pad, lpb, zero_pad), axis=0)
    # predict audio with the model
    predicted = model.predict_on_batch([ np.expand_dims(lpb, axis=0).astype(np.float32), np.expand_dims(mic, axis=0).astype(
        np.float32)])
    # squeeze the batch dimension away
    predicted_speech = np.squeeze(predicted)
    predicted_speech = predicted_speech[384:384+len_orig]
    sig[:,6] = predicted_speech # 2:128    3:256     4:512
    # write the file to target destination
    #sf.write(out_file_name.replace('mic.wav', 'out.wav'), sig, fs)
    sf.write(out_file_name, sig, fs)


def process_folder(model, folder_name, new_folder_name):
    '''
    批量处理目录中所有.wav文件
    Function to find .wav files in the folder and subfolders of "folder_name",
    process each .wav file with an algorithm and write it back to disk in the 
    folder "new_folder_name". The structure of the original directory is 
    preserved. The processed files will be saved with the same name as the 
    original file.

    Parameters
    ----------
    model : Keras model
        Keras model, which accepts audio in the size (1,timesteps).
    folder_name : STRING
        Input folder with .wav files.
    new_folder_name : STRING
        Traget folder for the processed files.

    '''

    # empty list for file and folder names
    file_names = []
    directories = []
    new_directories = []
    # walk through the directory
    for root, dirs, files in os.walk(folder_name):
        for file in files:
            # look for .wav files
            if file.endswith(".wav"): # "mic.wav" output_combine
                # write paths and filenames to lists
                file_names.append(file)
                directories.append(root)
                # create new directory names
                new_directories.append(root.replace(
                    folder_name, new_folder_name))
                # check if the new directory already exists, if not create it
                if not os.path.exists(root.replace(folder_name, new_folder_name)):
                    os.makedirs(root.replace(folder_name, new_folder_name))
    # iterate over all .wav files
    for idx in range(len(file_names)):
        # process each file with the model
        process_file(model, os.path.join(directories[idx], file_names[idx]),
                     os.path.join(new_directories[idx], file_names[idx]))
        print(file_names[idx] + ' processed successfully!')


if __name__ == '__main__':
    # arguement parser for running directly from the command line
    parser = argparse.ArgumentParser(description='data evaluation')
    parser.add_argument('--in_folder', '-i',
                        help='folder with input files')
    parser.add_argument('--out_folder', '-o',
                        help='target folder for processed files')
    parser.add_argument('--model', '-m',
                        help='weights of the enhancement model in .h5 format')
    args = parser.parse_args()
    # determine type of model
    if args.model.find('_norm_') != -1:
        norm_stft = True
    else:
        norm_stft = False

    # create class instance
    modelClass = DTLN_model()
    # build the model in default configuration
    modelClass.build_DTLN_model(norm_stft=True) # norm_stft=True
    # load weights of the .h5 file
    modelClass.model.load_weights(args.model)
    # process the folder
    process_folder(modelClass.model, args.in_folder, args.out_folder)
