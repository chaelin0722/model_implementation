import datetime
from tensorflow.keras.layers import Dropout, AveragePooling2D, Dense, Conv2D, MaxPooling2D, Activation, Concatenate, \
    GlobalAveragePooling2D, Flatten
from tensorflow.keras import Input
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import math
from tensorflow.keras.optimizers import Adam

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
    x_train = tf.image.resize(x_train, (299,299))
    x_train = x_train / 255

    return x_train, y_train


def transform_images(x_train):

    x_train = tf.image.resize(x_train, (299, 299))
    x_train = tf.image.random_crop(x_train, (299, 299, 3))
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
        dataset = dataset.shuffle(buffer_size=10)


    return dataset



def conv_block(x, num_filter, num_row, num_col, padding='same', strides=(1, 1), use_bias=False):

    x = Conv2D(num_filter, kernel_size=(num_row, num_col), strides=strides, padding=padding, use_bias=use_bias)(x)

    '''컨볼루션 레이어 설정 옵션에는 border_mode(경계 처리 방법)가 있는데, ‘valid’와 ‘same’으로 설정할 수 있습니다. 
    ‘valid’인 경우에는 입력 이미지 영역에 맞게 필터를 적용하기 때문에 출력 이미지 크기가 입력 이미지 크기보다 작아집니다. 
    반면에 ‘same’은 출력 이미지와 입력 이미지 사이즈가 동일하도록 입력 이미지 경계에 빈 영역을 추가하여 필터를 적용합니다. 
    ‘same’으로 설정 시, 입력 이미지에 경계를 학습시키는 효과가 있습니다.'''

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def stem(input):

    x = conv_block(input, 32, 3, 3, strides=(2,2), padding='valid')
    x = conv_block(x, 32, 3, 3, padding='valid')
    x = conv_block(x, 64, 3, 3)

    x1 = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid')(x)
    x2 = conv_block(x, 96, 3, 3, strides=(2,2), padding='valid')

    x = tf.keras.layers.concatenate([x1, x2])

    x_1 = conv_block(x, 64, 1, 1)
    x_1 = conv_block(x_1, 96, 3, 3, padding='valid')

    x_2 = conv_block(x, 64, 1, 1)
    x_2 = conv_block(x_2, 64, 7, 1)
    x_2 = conv_block(x_2, 64, 1, 7)
    x_2 = conv_block(x_2, 96, 3, 3, padding='valid')

    x = tf.keras.layers.concatenate([x_1, x_2])

    x__1 = conv_block(x, 192, 3, 3, strides=(2,2), padding='valid')
    x__2 = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid')(x)

    con = tf.keras.layers.concatenate([x__1, x__2], axis=-1)

    return con


def inception_a(input):

    a1 = conv_block(input, 64, 1, 1)
    a1 = conv_block(a1, 96, 3, 3)
    a1 = conv_block(a1, 96, 3, 3)

    a2 = conv_block(input, 64, 1, 1)
    a2 = conv_block(a2, 96, 3, 3)

    a3 = conv_block(input, 96, 1, 1)

    a4 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(input)
    a4 = conv_block(a4, 96, 1, 1)

    con = tf.keras.layers.concatenate([a1, a2, a3, a4])

    return con

def inception_b(input):

    b1 = conv_block(input, 192, 1, 1)
    b1 = conv_block(b1, 192, 1, 7)
    b1 = conv_block(b1, 224, 7, 1)
    b1 = conv_block(b1, 224, 1, 7)
    b1 = conv_block(b1, 256, 7, 1)

    b2 = conv_block(input, 192, 1, 1)
    b2 = conv_block(b2, 224, 1, 7)
    b2 = conv_block(b2, 256, 7, 1)

    b3 = conv_block(input, 384, 1, 1)

    b4 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(input)
    b4 = conv_block(b4, 128, 1, 1)

    con = tf.keras.layers.concatenate([b1, b2, b3, b4])

    return con

def inception_c(input):
    c1 = conv_block(input, 384, 1, 1)
    c1 = conv_block(c1, 448, 1, 3)
    c1 = conv_block(c1, 512, 3, 1)
    c1_1 = conv_block(c1, 256, 1, 3)
    c1_2 = conv_block(c1, 256, 3, 1)

    c2 = conv_block(input, 384, 1, 1)
    c2_1 = conv_block(c2, 256, 1, 3)
    c2_2 = conv_block(c2, 256, 3, 1)

    c3 = conv_block(input, 256, 1, 1)

    c4 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(input)
    c4 = conv_block(c4, 256, 1, 1)

    con = tf.keras.layers.concatenate([c1_1, c1_2, c2_1, c2_2, c3, c4])

    # 8x8, 1536
    return con

def reduction_a(input):

    r1 = conv_block(input, 192, 1, 1)
    r1 = conv_block(r1, 224, 3, 3)
    r1 = conv_block(r1, 256, 3, 3, strides=(2,2), padding='valid')

    r2 = conv_block(input, 384, 3, 3, strides=(2,2), padding='valid')

    r3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid')(input)

    con = tf.keras.layers.concatenate([r1, r2, r3])

    return con

def reduction_b(input):

    r1 = conv_block(input, 256, 1, 1)
    r1 = conv_block(r1, 256, 1, 7)
    r1 = conv_block(r1, 320, 7, 1)
    r1 = conv_block(r1, 320, 3, 3, strides=(2,2), padding='valid')

    r2 = conv_block(input, 192, 1, 1)
    r2 = conv_block(r2, 192, 3, 3, strides=(2,2), padding='valid')

    r3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(input)

    con = tf.keras.layers.concatenate([r1, r2, r3])

    return con



def create_inception_v4():

    input_data = Input(shape=(299, 299, 3))

    x = stem(input_data)

    # 4xinception-A
    for i in range(4):
        x = inception_a(x)

    x = reduction_a(x)

    for i in range(7):
        x = inception_b(x)

    x = reduction_b(x)

    for i in range(3):
        x = inception_c(x)

    x = AveragePooling2D(pool_size=(8, 8), padding='valid')(x)  #1536
    x = Dropout(0.2)(x)

    #    x = GlobalAveragePooling2D()(x)   #1536
    #    x = Dropout(0.8)(x)
    x = Flatten()(x)

    output = Dense(1000, activation='softmax')(x)

    model = Model(inputs=input_data, outputs=output, name='Inception-v4')

    return model


def main():

    train_url = "/home/ivpl-d14/PycharmProjects/pythonProject/model_implementation/Model-Implementation/tfrecords/tf_train/train2.tfrecord"
    val_url = "/home/ivpl-d14/PycharmProjects/pythonProject/model_implementation/Model-Implementation/tfrecords/tf_train/val2.tfrecord"

    train_dataset = load_tfrecord_dataset(train_url, 32, shuffle=True)
    val_dataset = load_tfrecord_dataset(val_url, 32, shuffle=False)

    # create inception-v4 model
    model = create_inception_v4()

    #model.summary()

    EPOCH = 100
    BATCH_SIZE = 32

    # decay every two epochs using exponential rate of 0.94
    def step_decay(epoch):
        init_lr = 0.045
        drop = 0.94
        epochs_drop = 2.0
        lrate = init_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        # to check on the tensorboard
        tf.summary.scalar('learning rate', data=lrate, step=epoch)

        return lrate

    # decayed everyh two epochs using exponential rate of 0.94
    rmsprop = tf.keras.optimizers.RMSprop(learning_rate=0.00001, decay=0.9, momentum=0.9, epsilon=1e-5)
    # sgd = tf.keras.optimizers.SGD(lr=1., decay=0.01, momentum=0.9, nesterov=True)
    sgd = tf.keras.optimizers.SGD(lr=0.0001, decay=0.9, momentum=0.9, nesterov=True)
    # adam = tf.keras.optimizers.Adam(learning_rate=0.01)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=step_decay(0)),
                  metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(5)])

    filename = 'checkpoints/checkpoint-epoch-{}-batch-{}-trial-001.h5'.format(EPOCH, BATCH_SIZE)
    log_dir = "./logs/scalar/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    file_writer.set_as_default()

    # 모델의 가중치를 저장하는 콜백
    callbacks = [

            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
            # 개선된 validation score를 도출해낼 때마다 weight를 중간 저장
            ModelCheckpoint(filepath=filename,
                             save_weights_only=True,
                             verbose=1,  # 로그를 출력
                             save_best_only=True,  # 가장 best 값만 저장
                             mode='auto'),  # auto는 알아서 best를 찾습니다. min/max

            tf.keras.callbacks.LearningRateScheduler(step_decay)
    ]

    steps_per_epoch = int(1231167/BATCH_SIZE)
    validation_steps = int(50000 /BATCH_SIZE)

    model.fit(train_dataset, validation_data=val_dataset, validation_steps=validation_steps,
               epochs=EPOCH, batch_size=BATCH_SIZE,  steps_per_epoch=steps_per_epoch, callbacks=callbacks)
                ##initial_epoch=100,
    #모델 저장하기
    model.save('my_inception_v4.h5')  #my_googLeNet_MORE_EPOCHS.h5


if __name__ == '__main__':
    main()


