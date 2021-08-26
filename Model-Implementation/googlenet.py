import tensorflow as tf
import datetime
import os

from tensorflow.keras import datasets, layers, models, losses, Model
from tensorflow.keras.layers import Dropout, AveragePooling2D, Dense, Conv2D, MaxPooling2D, Activation, Concatenate, \
    GlobalAveragePooling2D, Flatten
from tensorflow.keras import Input

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def inception(x_input, filter_1, filter_3_R, filter_3, filter_5_R, filter_5, pool_proj):
    x1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='SAME')(x_input)
    x1 = Conv2D(filters=pool_proj, kernel_size=(1, 1), padding="SAME")(x1)
    x1 = Activation('relu')(x1)

    # conv 1x1 -> conv 5x5 == conv 5x5 reduction -> conv 5x5
    x2 = Conv2D(filters=filter_5_R, kernel_size=(1, 1), padding="SAME")(x_input)
    x2 = Conv2D(filters=filter_5, kernel_size=(5, 5), padding="SAME")(x2)
    x2 = Activation('relu')(x2)

    x3 = Conv2D(filters=filter_3_R, kernel_size=(1, 1), padding="SAME")(x_input)
    x3 = Conv2D(filters=filter_3, kernel_size=(3, 3), padding="SAME")(x3)
    x3 = Activation('relu')(x3)

    x4 = Conv2D(filters=filter_1, kernel_size=(1, 1), padding="SAME")(x_input)
    x4 = Activation('relu')(x4)

    # COncatenate each layers' depth
    inception_result = Concatenate()([x1, x2, x3, x4])

    return inception_result


###

## INPUT = 244 X 244, RGB channel
input_data = Input(shape=(224, 224, 3))


################# PART 1 ######################

x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="SAME", activation='relu')(input_data)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="SAME")(x)
x = tf.keras.layers.LayerNormalization()(x)

x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="SAME", activation='relu')(x)
x = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation='relu')(x)

x = tf.keras.layers.LayerNormalization()(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="SAME")(x)

################# inception 3a, inception 3b ######################
x = inception(x, 64, 96, 128, 16, 32, 32)
x = inception(x, 128, 128, 192, 32, 96, 64)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="SAME")(x)

################# inception 4a ~ inception 4e ######################
x = inception(x, 192, 96, 208, 16, 48, 64)

# auxiliary classifier for over influencing
ax1 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(x)
ax1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="SAME", activation='relu')(ax1)
ax1 = Flatten()(ax1)
ax1 = Dense(1024, activation="relu")(ax1)  ## FC
ax1 = Dropout(0.7)(ax1)
ax1 = Dense(1000, activation="softmax")(ax1)  ## FC

x = inception(x, 160, 112, 224, 24, 64, 64)
x = inception(x, 128, 128, 256, 24, 64, 64)
x = inception(x, 112, 144, 288, 32, 64, 64)

# auxiliary classifier for over influencing
ax2 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(x)
ax2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="SAME", activation='relu')(ax2)
ax2 = Flatten()(ax2)
ax2 = Dense(1024, activation="relu")(ax2)  ## FC
ax2 = Dropout(0.7)(ax2)
ax2 = Dense(1000, activation="softmax")(ax2)  ## FC

x = inception(x, 256, 160, 320, 32, 128, 128)

x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="SAME")(x)

################# inception 5a, inceptino 5b ######################
x = inception(x, 256, 160, 320, 32, 128, 128)
x = inception(x, 384, 192, 384, 48, 128, 128)

x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)

outputs = Dense(1000, activation="softmax")(x)

googlenet = tf.keras.models.Model(inputs=input_data, outputs=[outputs, ax1, ax2], name='googlenet')
googlenet.summary()