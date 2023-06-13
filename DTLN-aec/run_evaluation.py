from DTLN_model import DTLN_model
import os
import librosa
import numpy as np
import soundfile as sf
'''
farend='samples/farend_speech_fileid_0.wav'
nearend='samples/nearend_mic_fileid_0.wav'

farend_speech,fs = librosa.core.load(farend, sr=16000, mono=True)
nearend_signal,fs = librosa.core.load(nearend, sr=16000, mono=True)

modelClass = DTLN_model();

modelClass.build_DTLN_model(norm_stft=True)

modelClass.model.load_weights('weights/DTLN_model.h5')

farend_speech2=np.expand_dims(farend_speech,axis=0)
nearend_signal2=np.expand_dims(nearend_signal,axis=0)

print('Model Running ....')
out=modelClass.model.predict_on_batch([farend_speech2,nearend_signal2])

out = np.squeeze(out)

sf.write('samples/predicted_speech.wav', out,fs)

print('Done')
'''
# use the GPU with idx 0
os.environ["CUDA_VISIBLE_DEVICES"]='1,2,0'
# activate this for some reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'

audio_path = '/data/projects/AEC-Challenge/mydatasets/aec_testset_v7/result/'
audio_names = os.listdir(audio_path)
for audio_name in audio_names:
    audio, fs = sf.read(audio_path + audio_name)
    mic=audio[:,3]
    lpb=audio[:,1]
    modelClass = DTLN_model();

    modelClass.build_DTLN_model() #norm_stft=True
    modelClass.model.load_weights('/home/ljl/projects/DTLN/DTLN-aec_v2/models_DTLN_model_v2_s_512/DTLN_model_v2_s_512.h5')
    # '/home/ljl/projects/DTLN/DTLN-aec_v2/models_DTLN_aec_v2_s_512_norm/DTLN_aec_v2_s_512_norm.h5'
    farend_speech2=np.expand_dims(lpb,axis=0)
    nearend_signal2=np.expand_dims(mic,axis=0)
    print('Model Running ....')
    out=modelClass.model.predict_on_batch([farend_speech2,nearend_signal2])

    out = np.squeeze(out)
    audio[:,5] = out  #3:pre_512 4:v2_s_512 

    sf.write(audio_path + audio_name, audio, fs)

    print('Done')

    
