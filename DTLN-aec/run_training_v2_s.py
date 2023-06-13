from DTLN_model import DTLN_model
import os

# use the GPU with idx 0
os.environ["CUDA_VISIBLE_DEVICES"]='1,0'
# activate this for some reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'




'''
# path to folder containing the noisy or mixed audio training files
path_to_train_mix = '/path/to/noisy/training/data/'
# path to folder containing the clean/speech files for training
path_to_train_speech = '/path/to/clean/training/data/'
# path to folder containing the noisy or mixed audio validation data
path_to_val_mix = '/path/to/noisy/validation/data/'
# path to folder containing the clean audio validation data
path_to_val_speech = '/path/to/clean/validation/data/'
'''
path_to_train_nearend_signal='/data/projects/AEC-Challenge/mydatasets/aec_noMute/v2_s_99h/train/nearend_mic' #'E:/intern/codes/dataset/nearend_mic_signal/'
path_to_train_farend_signal='/data/projects/AEC-Challenge/mydatasets/aec_noMute/v2_s_99h/train/farend_speech' #'E:/intern/codes/dataset/farend_signal/'
path_to_train_nearend_speech='/data/projects/AEC-Challenge/mydatasets/aec_noMute/v2_s_99h/train/nearend_speech' #'E:/intern/codes/dataset/nearend_speech/'

path_to_val_nearend_signal= '/data/projects/AEC-Challenge/mydatasets/aec_noMute/v2_s_99h/val/nearend_mic'
path_to_val_farend_signal= '/data/projects/AEC-Challenge/mydatasets/aec_noMute/v2_s_99h/val/farend_speech'
path_to_val_nearend_speech= '/data/projects/AEC-Challenge/mydatasets/aec_noMute/v2_s_99h/val/nearend_speech'

# name your training run
runName = 'DTLN_aec_v2_s_512_norm_99h'
# create instance of the DTLN model class
modelTrainer = DTLN_model()
# build the model
modelTrainer.build_DTLN_model(norm_stft=True) #norm_stft=True
# compile it with optimizer and cost function for training
modelTrainer.compile_model()
# train the model
modelTrainer.train_model(runName, path_to_train_nearend_signal, path_to_train_farend_signal, path_to_train_nearend_speech, \
                                  path_to_val_nearend_signal, path_to_val_farend_signal, path_to_val_nearend_speech)

