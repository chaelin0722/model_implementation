import tensorflow as tf
from tensorflow import keras
import tensorflow.keras
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import datetime
import numpy as np
import time
import os
from PIL import Image

## load data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
validation_images, validation_labels = train_images[:5000], train_labels[:5000]
train_images, train_labels = train_images[5000:], train_labels[5000:]

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

#preprocessing
plt.figure(figsize=(20,20))
for i, (image, label) in enumerate(train_ds.take(5)):
    ax = plt.subplot(5,5,i+1)
    plt.imshow(image)
    plt.title(CLASS_NAMES[label.numpy()[0]])
    plt.axis('off')


def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227, 227))
    return image, label

train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
print("Training data size:", train_ds_size)
print("Test data size:", test_ds_size)
print("Validation data size:", validation_ds_size)

train_ds = (train_ds.map(process_images).shuffle(buffer_size=train_ds_size).batch(batch_size=8, drop_remainder=True))
test_ds = (test_ds.map(process_images).shuffle(buffer_size=train_ds_size).batch(batch_size=8, drop_remainder=True))
validation_ds = (validation_ds.map(process_images).shuffle(buffer_size=train_ds_size).batch(batch_size=8, drop_remainder=True))


model = tf.keras.models.Sequential([
    # C1 (first layer)
    tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=4, activation='relu', input_shape=(227,227,3)),
    tf.keras.layers.BatchNormalization(), # currently use batch normalization instead local response normalization
    # overlapping
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid', data_format=None),

    # C2
    tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=1, activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(), # currently use batch normalization instead local response normalization
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid', data_format=None),

    # C3
    tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, activation='relu',padding="same"),
    tf.keras.layers.BatchNormalization(),
    # C4
    tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, activation='relu',padding="same"),
    tf.keras.layers.BatchNormalization(),
    # C5
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu',padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid', data_format=None),

    # Flatten!
    tf.keras.layers.Flatten(),
    # F6
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    # F7
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    # outputlayer, Softmax
    tf.keras.layers.Dense(10, activation='softmax')
])

# check tensorboard
root_logdir = os.path.join(os.curdir, "logs\\fit\\")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

## compile
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.optimizers.SGD(lr=0.001),
              metrics=['accuracy'])
#model.summary()

# start training
model.fit(train_ds,
          epochs=5,
          validation_data=validation_ds,
          validation_freq=1,
          callbacks=[tensorboard_cb])
