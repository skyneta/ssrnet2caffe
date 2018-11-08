import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

import keras2caffe
from SSRNET_model import SSR_net, SSR_net_general

if __name__ == '__main__':
    # model path
    weight_file = "ssrnet_3_3_3_64_1.0_1.0.h5"
    weight_file_gender = "ssrnet_3_3_3_64_1.0_1.0.h5"

    # load model and weights
    img_size = 64
    stage_num = [3, 3, 3]
    lambda_local = 1
    lambda_d = 1

    # load model
    model = SSR_net(img_size, stage_num, lambda_local, lambda_d)()
    model.load_weights(weight_file)
    print(model)

    # 开始转换
    keras2caffe.convert(model, 'ssrnet.prototxt', 'ssrnet.caffemodel')
    print("Done")

