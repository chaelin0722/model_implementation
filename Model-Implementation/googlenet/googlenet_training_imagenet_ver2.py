import datetime
from tensorflow.keras.layers import Dropout, AveragePooling2D, Dense, Conv2D, MaxPooling2D, Activation, Concatenate, \
    GlobalAveragePooling2D, Flatten
from tensorflow.keras import Input
import tensorflow as tf
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

def _parse_tfrecord():
    def parse_tfrecord(tfrecord):
        features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                    'image/filename': tf.io.FixedLenFeature([], tf.string),
                    'image/encoded': tf.io.FixedLenFeature([], tf.string)}
        x = tf.io.parse_single_example(tfrecord, features)

        x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
        y_train = tf.cast(x['image/source_id'], tf.int64)
        x_train = _transform_images()(x_train)
        y_train = _transform_targets(y_train)
        return (x_train, y_train), y_train
    return parse_tfrecord


def _transform_images():
    def transform_images(x_train):
        x_train = tf.image.resize(x_train, (224, 224))
        x_train = tf.image.random_crop(x_train, (224,224, 3))
        x_train = tf.image.random_flip_left_right(x_train)
        x_train = tf.image.random_saturation(x_train, 0.6, 1.4)
        x_train = tf.image.random_brightness(x_train, 0.4)
        x_train = x_train / 255
        return x_train
    return transform_images


def _transform_targets(y_train):
    return y_train

def load_tfrecord_dataset(tfrecord_name, batch_size, shuffle=True):
    """load dataset from tfrecord"""
    #raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
    raw_dataset = tf.data.Dataset.list_files(tfrecord_name)

    raw_dataset = raw_dataset.interleave(tf.data.TFRecordDataset,
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                 deterministic=False)

    raw_dataset = raw_dataset.repeat()


    dataset = raw_dataset.map(
        _parse_tfrecord(),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size=10)

    return dataset




## model layer
def inception(x_input, filter_1, filter_3_R, filter_3, filter_5_R, filter_5, pool_proj):
    x1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='SAME')(x_input)
    x1 = Conv2D(filters=pool_proj, kernel_size=(1, 1), padding="SAME", activation='relu')(x1)

    # conv 1x1 -> conv 5x5 == conv 5x5 reduction -> conv 5x5
    x2 = Conv2D(filters=filter_5_R, kernel_size=(1, 1), padding="SAME", activation='relu')(x_input)
    x2 = Conv2D(filters=filter_5, kernel_size=(5, 5), padding="SAME", activation='relu')(x2)

    x3 = Conv2D(filters=filter_3_R, kernel_size=(1, 1), padding="SAME", activation='relu')(x_input)
    x3 = Conv2D(filters=filter_3, kernel_size=(3, 3), padding="SAME", activation='relu')(x3)

    x4 = Conv2D(filters=filter_1, kernel_size=(1, 1), padding="SAME", activation='relu')(x_input)

    # Concatenate each layers' depth
    inception_result = Concatenate()([x1, x2, x3, x4])

    return inception_result



def main():

    train_url = "/home/ivpl-d14/PycharmProjects/pythonProject/model_implementation/Model-Implementation/tfrecords/tf_train/train.tfrecord"
    val_url = "/home/ivpl-d14/PycharmProjects/pythonProject/model_implementation/Model-Implementation/tfrecords/tf_train/val.tfrecord"

    train_dataset = load_tfrecord_dataset(train_url,64)
    val_dataset = load_tfrecord_dataset(val_url,64)



    ## INPUT = 244 X 244, RGB channel
    input_data = Input((224, 224, 3))

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
    ax1 = Dense(1000, activation="softmax", name='ax1')(ax1)  ## FC

    x = inception(x, 160, 112, 224, 24, 64, 64)
    x = inception(x, 128, 128, 256, 24, 64, 64)
    x = inception(x, 112, 144, 288, 32, 64, 64)

    # auxiliary classifier for over influencing
    ax2 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(x)
    ax2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="SAME", activation='relu')(ax2)
    ax2 = Flatten()(ax2)
    ax2 = Dense(1024, activation="relu")(ax2)  ## FC
    ax2 = Dropout(0.7)(ax2)
    ax2 = Dense(1000, activation="softmax", name='ax2')(ax2)  ## FC

    x = inception(x, 256, 160, 320, 32, 128, 128)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="SAME")(x)

    ################# inception 5a, inceptino 5b ######################
    x = inception(x, 256, 160, 320, 32, 128, 128)
    x = inception(x, 384, 192, 384, 48, 128, 128)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)

    outputs = Dense(1000, activation="softmax", name='main_classifier')(x)

    concat_output = tf.keras.layers.concatenate([outputs*1.0, ax1*0.3, ax2*0.3])

    model = tf.keras.models.Model(inputs=input_data, outputs=concat_output, name='googlenet')
  # model.summary()
    ###
    learning_rate = 0.0001
    momentum = 0.9

    #optimizer = tf.keras.optimizers.SGD(lr=learning_rate, momentum=momentum, nesterov=False)
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

    model.load_weights('checkpoints/checkpoint-epoch-100-batch-64-trial-002.h5') # epoch 14

    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(5)])

    EPOCH = 110  #EPOCH=100
    BATCH_SIZE = 64

    #filename = 'checkpoints/checkpoint-epoch-{}-batch-{}-trial-002.h5'.format(EPOCH, BATCH_SIZE)
    filename =  'checkpoints/checkpoint-epoch-{}-batch-{}-trial-003.h5'.format(EPOCH, BATCH_SIZE)
    log_dir = "./logs/fit/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


    # 모델의 가중치를 저장하는 콜백
    callbacks = [

            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
            # 개선된 validation score를 도출해낼 때마다 weight를 중간 저장
            ModelCheckpoint(filepath=filename,
                             save_weights_only=True,
                             verbose=1,  # 로그를 출력
                             save_best_only=True,  # 가장 best 값만 저장
                             mode='auto')  # auto는 알아서 best를 찾습니다. min/max
    ]


    steps_per_epoch = int(1231167/BATCH_SIZE)
    validation_steps = int(50000 /BATCH_SIZE)

    model.fit(train_dataset,  batch_size=BATCH_SIZE,validation_data=val_dataset, validation_steps=validation_steps,
              initial_epoch=100, epochs=EPOCH,  steps_per_epoch=steps_per_epoch, callbacks=callbacks)
                ##initial_epoch=100,
    #모델 저장하기
    model.save('my_googLeNet_MORE_EPOCHS.h5')  #my_googLeNet_MORE_EPOCHS.h5


if __name__ == '__main__':
    main()








