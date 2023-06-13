import soundfile as sf
import numpy as np
import tensorflow.lite as tflite
import time
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
block_len = 512
block_shift = 128

interpreter_1 = tflite.Interpreter(model_path='./models_DTLN_aec_v6_s_fix_250h_v2_512_norm_1.tflite') #models_DTLN_aec_v6_s_fix_250h_v2_512_norm_1
interpreter_1.allocate_tensors()
interpreter_2 = tflite.Interpreter(model_path='./models_DTLN_aec_v6_s_fix_250h_v2_512_norm_2.tflite')
interpreter_2.allocate_tensors()

input_details_1 = interpreter_1.get_input_details()
output_details_1 = interpreter_1.get_output_details()

input_details_2 = interpreter_2.get_input_details()
output_details_2 = interpreter_2.get_output_details()

states_1 = np.zeros(input_details_1[2]['shape']).astype('float32')
states_2 = np.zeros(input_details_2[2]['shape']).astype('float32')

farend,fs = sf.read('lpb_10s.wav')
nearend,fs = sf.read('mic_10s.wav')

if fs != 16000:
    raise ValueError('This model only supports 16k sampling rate.')

out_file = np.zeros((len(nearend)))

farend_in_buffer = np.zeros((block_len)).astype('float32')
nearend_in_buffer = np.zeros((block_len)).astype('float32')
out_buffer = np.zeros((block_len)).astype('float32')

num_blocks = (nearend.shape[0] - (block_len-block_shift)) // block_shift
time_array = []

for idx in range(num_blocks):
    start_time = time.time()
    # shift values and write to buffer
    farend_in_buffer[:-block_shift] = farend_in_buffer[block_shift:]
    farend_in_buffer[-block_shift:] = farend[idx*block_shift:(idx*block_shift)+block_shift]
    nearend_in_buffer[:-block_shift] = nearend_in_buffer[block_shift:]
    nearend_in_buffer[-block_shift:] = nearend[idx*block_shift:(idx*block_shift)+block_shift]
    # calculate fft of input block
    farend_in_block_fft = np.fft.rfft(np.squeeze(farend_in_buffer)).astype("complex64")
    nearend_in_block_fft = np.fft.rfft(np.squeeze(nearend_in_buffer)).astype("complex64")
    farend_in_mag = np.abs(farend_in_block_fft)
    nearend_in_mag = np.abs(nearend_in_block_fft)
    nearend_in_phase = np.angle(nearend_in_block_fft)
    # reshape magnitude to input dimensions
    farend_in_mag = np.reshape(farend_in_mag, (1,1,-1)).astype('float32')
    nearend_in_mag = np.reshape(nearend_in_mag, (1,1,-1)).astype('float32')
    # set tensors to the first model
    interpreter_1.set_tensor(input_details_1[0]['index'], farend_in_mag)
    interpreter_1.set_tensor(input_details_1[1]['index'], nearend_in_mag)
    interpreter_1.set_tensor(input_details_1[2]['index'], states_1)
    # run calculation 
    interpreter_1.invoke()
    # get the output of the first block
    out_mask = interpreter_1.get_tensor(output_details_1[0]['index']) 
    states_1 = interpreter_1.get_tensor(output_details_1[1]['index']) 
    # calculate the ifft
    estimated_complex = nearend_in_block_fft * out_mask
    estimated_block = np.fft.irfft(estimated_complex)
    # reshape the time domain block
    estimated_block = np.reshape(estimated_block, (1,1,-1)).astype('float32')
    # set tensors to the second block
    interpreter_2.set_tensor(input_details_2[0]['index'], estimated_block)
    interpreter_2.set_tensor(input_details_2[1]['index'], np.reshape(farend_in_buffer, (1,1,-1)).astype('float32'))
    interpreter_2.set_tensor(input_details_2[2]['index'], states_2)
    # run calculation
    interpreter_2.invoke()
    # get output tensors
    out_block = interpreter_2.get_tensor(output_details_2[0]['index']) 
    states_2 = interpreter_2.get_tensor(output_details_2[1]['index']) 
    # shift values and write to buffer
    out_buffer[:-block_shift] = out_buffer[block_shift:]
    out_buffer[-block_shift:] = np.zeros((block_shift))
    out_buffer  += np.squeeze(out_block)
    # write block to output file
    out_file[idx*block_shift:(idx*block_shift)+block_shift] = out_buffer[:block_shift]
    time_array.append(time.time()-start_time)

predicted_speech = out_file[
        (block_len - block_shift) : (block_len - block_shift) + len(nearend)
    ]

# write to .wav file 
sf.write('out_10s_v6_2_512.wav', out_file, fs) 
print('Processing Time [ms]:')
print(np.mean(np.stack(time_array))*1000)
print('Processing finished.')