
from DTLN_model import DTLN_model
import argparse
from pkg_resources import parse_version
import tensorflow as tf
import os

# use the GPU with idx 0
os.environ["CUDA_VISIBLE_DEVICES"]='0'
# activate this for some reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'

quantization = False#True
weights_file='/home/ljl/projects/DTLN/DTLN-aec_v2/models_DTLN_aec_v6_s_fix_250h_v2_256_norm/DTLN_aec_v6_s_fix_250h_v2_256_norm.h5'#'/home/ljl/projects/DTLN/DTLN-aec_v2/models_DTLN_aec_v6_s_fix_128_norm/DTLN_aec_v6_s_fix_128_norm.h5'
target_folder='/home/ljl/projects/DTLN/DTLN-aec_v2/models_DTLN_aec_v6_s_fix_250h_v2_512_norm'
converter = DTLN_model()
converter.create_tf_lite_model(weights_file, 
                                  target_folder,
                                  norm_stft=True) #False
                                  #use_dynamic_range_quant=bool(quantization))
                                  