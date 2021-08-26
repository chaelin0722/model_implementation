import datetime
from tensorflow.keras.layers import Dropout, AveragePooling2D, Dense, Conv2D, MaxPooling2D, Activation, Concatenate, \
    GlobalAveragePooling2D, Flatten, MaxPool2D
from tensorflow.keras import Input
import tensorflow as tf
from keras.optimizers import SGD
import random
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from tensorflow.keras import regularizers

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try: # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e: # Memory growth must be set before GPUs have been initialized
        print(e)


def parse_tfrecord(tfrecord):
    features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                'image/encoded': tf.io.FixedLenFeature([], tf.string)}
    x = tf.io.parse_single_example(tfrecord, features)

    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    x_train = transform_images(x_train)
    y_train = tf.cast(x['image/source_id'], tf.int64)
    y_train = _transform_targets(y_train)

    return x_train, y_train


def parse_tfrecord_no_transform(tfrecord):
    features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                'image/encoded': tf.io.FixedLenFeature([], tf.string)}

    x = tf.io.parse_single_example(tfrecord, features)

    y_train = tf.cast(x['image/source_id'], tf.int64)
    y_train = _transform_targets(y_train)
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    x_train = tf.image.resize(x_train, (224,224))
    x_train = x_train / 255

    return x_train, y_train


def transform_images(x_train):

    x_train = tf.image.resize(x_train, (224,224))
    x_train = tf.image.random_crop(x_train, (224,224, 3))
    x_train = tf.image.random_flip_left_right(x_train)
    x_train = tf.image.random_saturation(x_train, 0.6, 1.4)
    x_train = tf.image.random_brightness(x_train, 0.4)
    x_train = x_train / 255

    return x_train


def _transform_targets(y_train):
    return y_train

def load_tfrecord_dataset(tfrecord_name, batch_size, shuffle):

    """load dataset from tfrecord"""
    raw_dataset = tf.data.Dataset.list_files(tfrecord_name, seed=42)
    raw_dataset = tf.data.TFRecordDataset(raw_dataset)

    '''
    raw_dataset = raw_dataset.interleave(tf.data.TFRecordDataset,
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                 deterministic=False)
    '''

    if shuffle is True:
        dataset = raw_dataset.map(
            parse_tfrecord,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    elif shuffle is False:
        dataset = raw_dataset.map(
            parse_tfrecord_no_transform,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=1024)


    return dataset

def main():

    input_data = Input(shape=(224, 224, 3))

    x = Conv2D(filters=64, kernel_size=(3, 3), padding="SAME", activation='relu',kernel_regularizer=regularizers.l2(0.0005))(input_data)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding="SAME", activation='relu',kernel_regularizer=regularizers.l2(0.0005))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), padding="SAME", activation='relu',kernel_regularizer=regularizers.l2(0.0005))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding="SAME", activation='relu',kernel_regularizer=regularizers.l2(0.0005))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)

    x = Conv2D(filters=256, kernel_size=(3, 3), padding="SAME", activation='relu',kernel_regularizer=regularizers.l2(0.0005))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding="SAME", activation='relu',kernel_regularizer=regularizers.l2(0.0005))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding="SAME", activation='relu',kernel_regularizer=regularizers.l2(0.0005))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(filters=512, kernel_size=(3, 3), padding="SAME", activation='relu',kernel_regularizer=regularizers.l2(0.0005))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding="SAME", activation='relu',kernel_regularizer=regularizers.l2(0.0005))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding="SAME", activation='relu',kernel_regularizer=regularizers.l2(0.0005))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)


    x = Conv2D(filters=512, kernel_size=(3, 3), padding="SAME", activation='relu',kernel_regularizer=regularizers.l2(0.0005))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding="SAME", activation='relu',kernel_regularizer=regularizers.l2(0.0005))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding="SAME", activation='relu',kernel_regularizer=regularizers.l2(0.0005))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(x)  ## FC
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(x)
    x = Dropout(0.5)(x)

    output = Dense(1000, activation="softmax")(x) #, kernel_regularizer=regularizers.l2(0.001))(x)

    model = Model(inputs=input_data, outputs=output, name='vgg-16')
    #model.summary()


    train_url = "/home/ivpl-d14/PycharmProjects/pythonProject/model_implementation/Model-Implementation/tfrecords/tf_train/train2.tfrecord"
    val_url = "/home/ivpl-d14/PycharmProjects/pythonProject/model_implementation/Model-Implementation/tfrecords/tf_train/val2.tfrecord"
    train_dataset = load_tfrecord_dataset(train_url,16, shuffle=True)
    val_dataset = load_tfrecord_dataset(val_url, 16, shuffle=False)


    EPOCH = 100
    BATCH_SIZE = 16

    adam = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=0.9)
    # sgd = tf.keras.optimizers.SGD(lr=0.00001, decay=0.01, momentum=0.9, nesterov=True)
    sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy',
                  metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(5)])

    filename = 'checkpoints/checkpoint-epoch-{}-batch-{}-trial-001.h5'.format(EPOCH, BATCH_SIZE)
    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # 모델의 가중치를 저장하는 콜백
    callbacks = [

        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        # 개선된 validation score를 도출해낼 때마다 weight를 중간 저장
        ModelCheckpoint(filepath=filename,
                        save_weights_only=True,
                        verbose=1,  # 로그를 출력
                        save_best_only=True,  # 가장 best 값만 저장
                        mode='auto'),  # auto는 알아서 best를 찾습니다. min/max
    ]

    steps_per_epoch = int(1231167 / BATCH_SIZE)
    validation_steps = int(50000 / BATCH_SIZE)


    '''
    model.fit(train_dataset, validation_data = val_dataset,
              validation_steps=validation_steps,
              epochs=EPOCH, batch_size=BATCH_SIZE,
              steps_per_epoch=steps_per_epoch,
              callbacks=callbacks)

    # 모델 저장
    model.save('my_vgg_16.h5')
    '''
    model.summary()

if __name__ == '__main__':
    main()
