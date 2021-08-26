import datetime
from tensorflow.keras.layers import Dropout, AveragePooling2D, Dense, Conv2D, MaxPooling2D, Activation, Concatenate, \
    GlobalAveragePooling2D, Flatten
from tensorflow.keras import Input
import tensorflow as tf
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

