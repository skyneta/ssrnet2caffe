import logging
import sys
import numpy as np
from keras.models import Model
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout, Multiply, Embedding, Lambda, Add, Concatenate, Activation
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.utils import plot_model
from keras.engine.topology import Layer
from keras import activations, initializers, regularizers, constraints

sys.setrecursionlimit(2 ** 20)
np.random.seed(2 ** 10)


class SSR_net:
    def __init__(self, image_size,stage_num,lambda_local,lambda_d):
        
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)


        self.stage_num = stage_num
        self.lambda_local = lambda_local
        self.lambda_d = lambda_d

#    def create_model(self):
    def __call__(self):
        logging.debug("Creating model...")


        inputs = Input(shape=self._input_shape)

        #-------------------------------------------------------------------------------------------------------------------------
        x = Conv2D(32,(3,3))(inputs)
        x = BatchNormalization(axis=self._channel_axis)(x)
        x = Activation('relu')(x)
        x_layer1 = AveragePooling2D(2,2)(x)
        x = Conv2D(32,(3,3))(x_layer1)
        x = BatchNormalization(axis=self._channel_axis)(x)
        x = Activation('relu')(x)
        x_layer2 = AveragePooling2D(2,2)(x)
        x = Conv2D(32,(3,3))(x_layer2)
        x = BatchNormalization(axis=self._channel_axis)(x)
        x = Activation('relu')(x)
        x_layer3 = AveragePooling2D(2,2)(x)
        x = Conv2D(32,(3,3))(x_layer3)
        x = BatchNormalization(axis=self._channel_axis)(x)
        x = Activation('relu')(x)
        #-------------------------------------------------------------------------------------------------------------------------
        s = Conv2D(16,(3,3))(inputs)
        s = BatchNormalization(axis=self._channel_axis)(s)
        s = Activation('tanh')(s)
        s_layer1 = MaxPooling2D(2,2)(s)
        s = Conv2D(16,(3,3))(s_layer1)
        s = BatchNormalization(axis=self._channel_axis)(s)
        s = Activation('tanh')(s)
        s_layer2 = MaxPooling2D(2,2)(s)
        s = Conv2D(16,(3,3))(s_layer2)
        s = BatchNormalization(axis=self._channel_axis)(s)
        s = Activation('tanh')(s)
        s_layer3 = MaxPooling2D(2,2)(s)
        s = Conv2D(16,(3,3))(s_layer3)
        s = BatchNormalization(axis=self._channel_axis)(s)
        s = Activation('tanh')(s)
        

        #-------------------------------------------------------------------------------------------------------------------------
        # Classifier block
        s_layer4 = Conv2D(10,(1,1),activation='relu')(s)
        s_layer4 = Flatten()(s_layer4)
        s_layer4_mix = Dropout(0.2)(s_layer4)
        s_layer4_mix = Dense(units=self.stage_num[0], activation="relu")(s_layer4_mix)
        
        x_layer4 = Conv2D(10,(1,1),activation='relu')(x)
        x_layer4 = Flatten()(x_layer4)
        x_layer4_mix = Dropout(0.2)(x_layer4)
        x_layer4_mix = Dense(units=self.stage_num[0], activation="relu")(x_layer4_mix)
        
        feat_a_s1_pre = Multiply()([s_layer4,x_layer4])
        delta_s1 = Dense(1,activation='tanh',name='delta_s1')(feat_a_s1_pre)
        
        feat_a_s1 = Multiply()([s_layer4_mix,x_layer4_mix])
        feat_a_s1 = Dense(2*self.stage_num[0],activation='relu')(feat_a_s1)
        pred_a_s1 = Dense(units=self.stage_num[0], activation="relu",name='pred_age_stage1')(feat_a_s1)
        #feat_local_s1 = Lambda(lambda x: x/10)(feat_a_s1)
        #feat_a_s1_local = Dropout(0.2)(pred_a_s1)
        local_s1 = Dense(units=self.stage_num[0], activation='tanh', name='local_delta_stage1')(feat_a_s1)
        #-------------------------------------------------------------------------------------------------------------------------
        s_layer2 = Conv2D(10,(1,1),activation='relu')(s_layer2)
        s_layer2 = MaxPooling2D(4,4)(s_layer2)
        s_layer2 = Flatten()(s_layer2)
        s_layer2_mix = Dropout(0.2)(s_layer2)
        s_layer2_mix = Dense(self.stage_num[1],activation='relu')(s_layer2_mix)
        
        x_layer2 = Conv2D(10,(1,1),activation='relu')(x_layer2)
        x_layer2 = AveragePooling2D(4,4)(x_layer2)
        x_layer2 = Flatten()(x_layer2)
        x_layer2_mix = Dropout(0.2)(x_layer2)
        x_layer2_mix = Dense(self.stage_num[1],activation='relu')(x_layer2_mix)
        
        feat_a_s2_pre = Multiply()([s_layer2,x_layer2])
        delta_s2 = Dense(1,activation='tanh',name='delta_s2')(feat_a_s2_pre)
        
        feat_a_s2 = Multiply()([s_layer2_mix,x_layer2_mix])
        feat_a_s2 = Dense(2*self.stage_num[1],activation='relu')(feat_a_s2)
        pred_a_s2 = Dense(units=self.stage_num[1], activation="relu",name='pred_age_stage2')(feat_a_s2)
        #feat_local_s2 = Lambda(lambda x: x/10)(feat_a_s2)
        #feat_a_s2_local = Dropout(0.2)(pred_a_s2)
        local_s2 = Dense(units=self.stage_num[1], activation='tanh', name='local_delta_stage2')(feat_a_s2)
        #-------------------------------------------------------------------------------------------------------------------------
        s_layer1 = Conv2D(10,(1,1),activation='relu')(s_layer1)
        s_layer1 = MaxPooling2D(8,8)(s_layer1)
        s_layer1 = Flatten()(s_layer1)
        s_layer1_mix = Dropout(0.2)(s_layer1)
        s_layer1_mix = Dense(self.stage_num[2],activation='relu')(s_layer1_mix)
        
        x_layer1 = Conv2D(10,(1,1),activation='relu')(x_layer1)
        x_layer1 = AveragePooling2D(8,8)(x_layer1)
        x_layer1 = Flatten()(x_layer1)
        x_layer1_mix = Dropout(0.2)(x_layer1)
        x_layer1_mix = Dense(self.stage_num[2],activation='relu')(x_layer1_mix)

        feat_a_s3_pre = Multiply()([s_layer1,x_layer1])
        delta_s3 = Dense(1,activation='tanh',name='delta_s3')(feat_a_s3_pre)
        
        feat_a_s3 = Multiply()([s_layer1_mix,x_layer1_mix])
        feat_a_s3 = Dense(2*self.stage_num[2],activation='relu')(feat_a_s3)
        pred_a_s3 = Dense(units=self.stage_num[2], activation="relu",name='pred_age_stage3')(feat_a_s3)
        #feat_local_s3 = Lambda(lambda x: x/10)(feat_a_s3)
        #feat_a_s3_local = Dropout(0.2)(pred_a_s3)
        local_s3 = Dense(units=self.stage_num[2], activation='tanh', name='local_delta_stage3')(feat_a_s3)
        #-------------------------------------------------------------------------------------------------------------------------

        pred_age = [pred_a_s1, pred_a_s2, pred_a_s3, delta_s1, delta_s2, delta_s3, local_s1, local_s2, local_s3]
        print(len(pred_age))
        model_age = Model(inputs=inputs, outputs=pred_age)
        print("model_age is ok")
        return model_age

class SSR_net_general:
    def __init__(self, image_size, stage_num, lambda_local, lambda_d):
        
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)


        self.stage_num = stage_num
        self.lambda_local = lambda_local
        self.lambda_d = lambda_d

#    def create_model(self):
    def __call__(self):
        logging.debug("Creating model...")


        inputs = Input(shape=self._input_shape)

        #-------------------------------------------------------------------------------------------------------------------------
        x = Conv2D(32,(3,3))(inputs)
        x = BatchNormalization(axis=self._channel_axis)(x)
        x = Activation('relu')(x)
        x_layer1 = AveragePooling2D(2,2)(x)
        x = Conv2D(32,(3,3))(x_layer1)
        x = BatchNormalization(axis=self._channel_axis)(x)
        x = Activation('relu')(x)
        x_layer2 = AveragePooling2D(2,2)(x)
        x = Conv2D(32,(3,3))(x_layer2)
        x = BatchNormalization(axis=self._channel_axis)(x)
        x = Activation('relu')(x)
        x_layer3 = AveragePooling2D(2,2)(x)
        x = Conv2D(32,(3,3))(x_layer3)
        x = BatchNormalization(axis=self._channel_axis)(x)
        x = Activation('relu')(x)
        #-------------------------------------------------------------------------------------------------------------------------
        s = Conv2D(16,(3,3))(inputs)
        s = BatchNormalization(axis=self._channel_axis)(s)
        s = Activation('tanh')(s)
        s_layer1 = MaxPooling2D(2,2)(s)
        s = Conv2D(16,(3,3))(s_layer1)
        s = BatchNormalization(axis=self._channel_axis)(s)
        s = Activation('tanh')(s)
        s_layer2 = MaxPooling2D(2,2)(s)
        s = Conv2D(16,(3,3))(s_layer2)
        s = BatchNormalization(axis=self._channel_axis)(s)
        s = Activation('tanh')(s)
        s_layer3 = MaxPooling2D(2,2)(s)
        s = Conv2D(16,(3,3))(s_layer3)
        s = BatchNormalization(axis=self._channel_axis)(s)
        s = Activation('tanh')(s)
        

        #-------------------------------------------------------------------------------------------------------------------------
        # Classifier block
        s_layer4 = Conv2D(10,(1,1),activation='relu')(s)
        s_layer4 = Flatten()(s_layer4)
        s_layer4_mix = Dropout(0.2)(s_layer4)
        s_layer4_mix = Dense(units=self.stage_num[0], activation="relu")(s_layer4_mix)
        
        x_layer4 = Conv2D(10,(1,1),activation='relu')(x)
        x_layer4 = Flatten()(x_layer4)
        x_layer4_mix = Dropout(0.2)(x_layer4)
        x_layer4_mix = Dense(units=self.stage_num[0], activation="relu")(x_layer4_mix)
        
        feat_s1_pre = Multiply()([s_layer4,x_layer4])
        delta_s1 = Dense(1,activation='tanh',name='delta_s1')(feat_s1_pre)
        
        feat_s1 = Multiply()([s_layer4_mix,x_layer4_mix])
        feat_s1 = Dense(2*self.stage_num[0],activation='relu')(feat_s1)
        pred_s1 = Dense(units=self.stage_num[0], activation="relu",name='pred_stage1')(feat_s1)
        local_s1 = Dense(units=self.stage_num[0], activation='tanh', name='local_delta_stage1')(feat_s1)
        #-------------------------------------------------------------------------------------------------------------------------
        s_layer2 = Conv2D(10,(1,1),activation='relu')(s_layer2)
        s_layer2 = MaxPooling2D(4,4)(s_layer2)
        s_layer2 = Flatten()(s_layer2)
        s_layer2_mix = Dropout(0.2)(s_layer2)
        s_layer2_mix = Dense(self.stage_num[1],activation='relu')(s_layer2_mix)
        
        x_layer2 = Conv2D(10,(1,1),activation='relu')(x_layer2)
        x_layer2 = AveragePooling2D(4,4)(x_layer2)
        x_layer2 = Flatten()(x_layer2)
        x_layer2_mix = Dropout(0.2)(x_layer2)
        x_layer2_mix = Dense(self.stage_num[1],activation='relu')(x_layer2_mix)
        
        feat_s2_pre = Multiply()([s_layer2,x_layer2])
        delta_s2 = Dense(1,activation='tanh',name='delta_s2')(feat_s2_pre)
        
        feat_s2 = Multiply()([s_layer2_mix,x_layer2_mix])
        feat_s2 = Dense(2*self.stage_num[1],activation='relu')(feat_s2)
        pred_s2 = Dense(units=self.stage_num[1], activation="relu",name='pred_stage2')(feat_s2)
        local_s2 = Dense(units=self.stage_num[1], activation='tanh', name='local_delta_stage2')(feat_s2)
        #-------------------------------------------------------------------------------------------------------------------------
        s_layer1 = Conv2D(10,(1,1),activation='relu')(s_layer1)
        s_layer1 = MaxPooling2D(8,8)(s_layer1)
        s_layer1 = Flatten()(s_layer1)
        s_layer1_mix = Dropout(0.2)(s_layer1)
        s_layer1_mix = Dense(self.stage_num[2],activation='relu')(s_layer1_mix)
        
        x_layer1 = Conv2D(10,(1,1),activation='relu')(x_layer1)
        x_layer1 = AveragePooling2D(8,8)(x_layer1)
        x_layer1 = Flatten()(x_layer1)
        x_layer1_mix = Dropout(0.2)(x_layer1)
        x_layer1_mix = Dense(self.stage_num[2],activation='relu')(x_layer1_mix)

        feat_s3_pre = Multiply()([s_layer1,x_layer1])
        delta_s3 = Dense(1,activation='tanh',name='delta_s3')(feat_s3_pre)
        
        feat_s3 = Multiply()([s_layer1_mix,x_layer1_mix])
        feat_s3 = Dense(2*self.stage_num[2],activation='relu')(feat_s3)
        pred_s3 = Dense(units=self.stage_num[2], activation="relu",name='pred_stage3')(feat_s3)
        local_s3 = Dense(units=self.stage_num[2], activation='tanh', name='local_delta_stage3')(feat_s3)
        #-------------------------------------------------------------------------------------------------------------------------

        pred_general = [pred_s1, pred_s2, pred_s3, delta_s1, delta_s2, delta_s3, local_s1, local_s2, local_s3]
        print(len(pred_general))
        model_general = Model(inputs=inputs, outputs=pred_general)
        print("model_general is ok")
        return model_general