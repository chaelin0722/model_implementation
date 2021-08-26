import datetime
from tensorflow.keras.layers import Dropout, AveragePooling2D, Dense, Conv2D, MaxPooling2D, Activation, Concatenate, \
    GlobalAveragePooling2D, Flatten
from tensorflow.keras import Input
import tensorflow as tf
from keras.optimizers import SGD
import random
from keras.callbacks import ModelCheckpoint
from keras.models import Model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:  # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e:  # Memory growth must be set before GPUs have been initialized
        print(e)


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
        ran_image_size = random.randint(256, 513)  # min 256 ~ 512 max
        x_train = tf.image.resize(x_train, (ran_image_size, ran_image_size))
        x_train = tf.image.random_crop(x_train, (224, 224, 3))
        x_train = x_train / 255
        return x_train

    return transform_images


def _transform_targets(y_train):
    return y_train


def load_tfrecord_dataset(tfrecord_name, batch_size, shuffle=True):
    """load dataset from tfrecord"""
    # raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
    raw_dataset = tf.data.Dataset.list_files(tfrecord_name)

    raw_dataset = raw_dataset.interleave(tf.data.TFRecordDataset,
                                         num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                         deterministic=False)

    raw_dataset = raw_dataset.repeat()

    if shuffle:
        raw_dataset = raw_dataset.shuffle(buffer_size=1024)

    dataset = raw_dataset.map(
        _parse_tfrecord(),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def create_vgg_16():
    input_data = Input(shape=(224, 224, 3))

    x = Conv2D(filters=64, kernel_size=(3, 3), padding="SAME", activation='relu')(input_data)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding="SAME", activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), padding="SAME", activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding="SAME", activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)

    x = Conv2D(filters=256, kernel_size=(3, 3), padding="SAME", activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding="SAME", activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding="SAME", activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(filters=512, kernel_size=(3, 3), padding="SAME", activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding="SAME", activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding="SAME", activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(filters=512, kernel_size=(3, 3), padding="SAME", activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding="SAME", activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding="SAME", activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(4096, activation="relu")(x)  ## FC
    x = Dropout(0.5)(x)
    x = Dense(4096, activation="relu")(x)  ## FC
    x = Dropout(0.5)(x)

    output = Dense(1000, activation="softmax")(x)

    model = Model(inputs=input_data, outputs=output, name='vgg-16')

    return model


def main():
    train_url = "/home/ivpl-d14/PycharmProjects/pythonProject/model_implementation/Model-Implementation/tfrecords/tf_train/train.tfrecord"
    val_url = "/home/ivpl-d14/PycharmProjects/pythonProject/model_implementation/Model-Implementation/tfrecords/tf_train/val.tfrecord"

    train_dataset = load_tfrecord_dataset(train_url, 64)
    val_dataset = load_tfrecord_dataset(val_url, 64)

    model = create_vgg_16()
    # model.summary()

    EPOCH = 100
    BATCH_SIZE = 32

    adam = tf.keras.optimizers.Adam(learning_rate=0.00001, decay=0.9)
    # sgd = tf.keras.optimizers.SGD(lr=0.00001, decay=0.01, momentum=0.9, nesterov=True)
    # sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',
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

        # tf.keras.callbacks.LearningRateScheduler(step_decay)
    ]

    steps_per_epoch = int(1231167 / BATCH_SIZE)
    validation_steps = int(50000 / BATCH_SIZE)

    model.fit(train_dataset, validation_data=val_dataset, validation_steps=validation_steps,
              epochs=EPOCH, batch_size=BATCH_SIZE, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
    ##initial_epoch=100,

    # 모델 저장
    model.save('my_vgg_16.h5')


if __name__ == '__main__':
    main()
