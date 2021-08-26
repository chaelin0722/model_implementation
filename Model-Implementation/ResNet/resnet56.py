# https://keras.io/zh/examples/cifar10_resnet/

from tensorflow.keras.layers import AveragePooling2D, Dense, Conv2D, Activation, \
     Flatten, BatchNormalization
from tensorflow.keras import Input
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from tensorflow.keras import regularizers
from keras.layers import Add
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import datetime
import math

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def create_resnet50(depth, num_classes):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6) #3

    input_data = Input(shape=(32,32,3))

    x = resnet_layer(inputs = input_data)

    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides=1
            if stack > 0 and res_block == 0:
                strides = 2
            y = resnet_layer(inputs = x, num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs =y, num_filters=num_filters,
                             activation=None)

            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)

            x = Add()([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    #####
    x = AveragePooling2D((8,8))(x)
    x = Flatten()(x)
    output = Dense(10, activation='softmax', kernel_initializer='he_normal')(x)

    model = Model(input_data, outputs=output, name='resnet-50')

    return model



def main():
    # Load Cifar-10 data-set
    (train_im, train_lab), (test_im, test_lab) = tf.keras.datasets.cifar10.load_data()

    #### Normalize the images to pixel values (0, 1)
    train_im, test_im = train_im / 255.0, test_im / 255.0
    #### Check the format
    # print("train_im, train_lab
    # print("shape of images and labels array: ", train_im.shape, train_lab.shape)  # 50000 train data
    # print("shape of images and labels array ; test: ", test_im.shape, test_lab.shape) # 10000 test data

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    ### One hot encoding for labels
    train_lab_categorical = tf.keras.utils.to_categorical(train_lab, num_classes=10, dtype='uint8')

    train_im, valid_im, train_lab, valid_lab = train_test_split(train_im, train_lab_categorical,
                                                                test_size=0.20,
                                                                stratify=train_lab_categorical,
                                                                random_state=40, shuffle=True)

    # print("train data shape after the split: ", train_im.shape)  # after split, train data : 40000
    # print('new validation data shape: ', valid_im.shape)        # val data : 10000

    BATCH_SIZE = 128  # original=128
    EPOCH = 100  #200

    train_DataGen = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.2,
                                                                    width_shift_range=0.1,
                                                                    height_shift_range=0.1,
                                                                    horizontal_flip=True)

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    train_set_conv = train_DataGen.flow(train_im, train_lab, batch_size=BATCH_SIZE)  # train_lab is categorical same shape of (x_train, y_train)
    valid_set_conv = valid_datagen.flow(valid_im, valid_lab, batch_size=BATCH_SIZE)  # so as valid_lab

    ## decay learning rate
    def lrdecay(epoch):
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1

        tf.summary.scalar('learning rate', data=lr, step=epoch)

        return lr


    filename = 'checkpoints/checkpoint-epoch-{}-batch-{}-trial-001.h5'.format(EPOCH, BATCH_SIZE)
    log_dir = "./logs2/scalars/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    file_writer.set_as_default()


    # 콜백
    callbacks = [

            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
            tf.keras.callbacks.LearningRateScheduler(lrdecay),

            ModelCheckpoint(filepath=filename,
                             save_weights_only=True,
                             verbose=1,
                             save_best_only=True,
                             mode='auto')
    ]

    depth = 56

    model = create_resnet50(depth=depth, num_classes=10)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lrdecay(0)), metrics=['acc'])


    model.fit(train_set_conv,
                          batch_size=BATCH_SIZE,
                          epochs=EPOCH,
                          steps_per_epoch=int(train_im.shape[0]/BATCH_SIZE),  # shape[0] = row
                          validation_steps=int(valid_im.shape[0]/BATCH_SIZE),
                          validation_data=valid_set_conv,
                          callbacks=callbacks)

    model.save('my_resnet_56_ver2.h5')

    scores = model.evaluate(train_set_conv, verbose=1)
    print("Train loss: ", scores[0])
    print("Train accuracy: ", scores[1])


if __name__ == '__main__':
    main()