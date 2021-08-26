
from tensorflow.keras.layers import AveragePooling2D, Dense, Conv2D, MaxPooling2D, Activation, Concatenate, \
    GlobalAveragePooling2D, Flatten, BatchNormalization, ZeroPadding2D
import matplotlib.pyplot as plt
from tensorflow.keras import Input
import tensorflow as tf
from keras.optimizers import SGD
import random
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from tensorflow.keras import regularizers
from keras.layers import Add
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import datetime

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try: # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e: # Memory growth must be set before GPUs have been initialized
        print(e)


# renet block where dimension does not change.
def res_identity(x, filters): # will have 3 blocks and then input will be added

  x_skip = x # this will be used for addition with the residual block
  f1, f2 = filters

  #first block
  x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=regularizers.l2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  #second block # bottleneck (but size kept same with padding)
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  # third block activation used after adding the input
  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=regularizers.l2(0.001))(x)
  x = BatchNormalization()(x)
  # x = Activation('relu')(x)

  # add the input
  x = Add()([x, x_skip])
  x = Activation('relu')(x)

  return x


def res_conv(x, s, filters):
  x_skip = x
  f1, f2 = filters

  # first block
  x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=regularizers.l2(0.001))(x)
  # when s = 2 then it is like downsizing the feature map
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  # second block
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  #third block
  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=regularizers.l2(0.001))(x)
  x = BatchNormalization()(x)

  # shortcut
  x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=regularizers.l2(0.001))(x_skip)
  x_skip = BatchNormalization()(x_skip)

  # add
  x = Add()([x, x_skip])
  x = Activation('relu')(x)

  return x


def create_resnet50():
    input_data = Input(shape=(32,32,3))

    ## conv 1
    x = ZeroPadding2D(padding=(3,3))(input_data)
    x = Conv2D(64, kernel_size=(7,7), strides=(2,2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    x = MaxPooling2D((3,3), strides=(2,2))(x)

    ## conv 2_x
    x = res_conv(x, s=1, filters=(64, 256))
    x = res_identity(x, filters=(64,256))
    x = res_identity(x, filters=(64,256))

    ## conv 3_x
    x = res_conv(x, s=2, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))

    ## conv 4_x
    x = res_conv(x, s=2, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))

    ## conv 5_x
    x = res_conv(x, s=2, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))

    #####
    x = GlobalAveragePooling2D((2,2), padding='SAME')(x)
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
    # print("train_im, train_lab types: ", type(train_im), type(train_lab))
    #### check the shape
    # print("shape of images and labels array: ", train_im.shape, train_lab.shape)  # 50000 train data
    # print("shape of images and labels array ; test: ", test_im.shape, test_lab.shape) # 10000 test data

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    ### One hot encoding for labels
    train_lab_categorical = tf.keras.utils.to_categorical(train_lab, num_classes=10, dtype='uint8')
    test_lab_categorical = tf.keras.utils.to_categorical(test_lab, num_classes=10, dtype='uint8')

    train_im, valid_im, train_lab, valid_lab = train_test_split(train_im, train_lab_categorical,
                                                                test_size=0.20,
                                                                stratify=train_lab_categorical,
                                                                random_state=40, shuffle=True)

    # print("train data shape after the split: ", train_im.shape)  # after split, train data : 40000
    # print('new validation data shape: ', valid_im.shape)        # val data : 10000

    BATCH_SIZE = 64
    EPOCH = 160

    train_DataGen = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.2,
                                                                    width_shift_range=0.1,
                                                                    height_shift_range=0.1,
                                                                    horizontal_flip=True)

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    train_set_conv = train_DataGen.flow(train_im, train_lab, batch_size=BATCH_SIZE)  # train_lab is categorical
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

        return lr

    filename = 'checkpoints/checkpoint-epoch-{}-batch-{}-trial-001.h5'.format(EPOCH, BATCH_SIZE)
    log_dir = "./logs/fit/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # 모델의 가중치를 저장하는 콜백
    callbacks = [

            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
            # 개선된 validation score를 도출해낼 때마다 weight를 중간 저장
            ModelCheckpoint(filepath=filename,
                             save_weights_only=True,
                             verbose=1,  # 로그를 출력
                             save_best_only=True,  # 가장 best 값만 저장
                             mode='auto'), # auto는 알아서 best를 찾습니다. min/max
            tf.keras.callbacks.LearningRateScheduler(lrdecay)
    ]

    model = create_resnet50()

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['acc'])


    model.fit(train_set_conv,
                          epochs=EPOCH,
                          steps_per_epoch=train_im.shape[0]/BATCH_SIZE,  # shape[0] = row
                          validation_steps=valid_im.shape[0]/BATCH_SIZE,
                          validation_data=valid_set_conv,
                          callbacks=callbacks)


    model.save('my_resnet_50.h5')



if __name__ == '__main__':
    main()