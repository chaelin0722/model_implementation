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
    try: # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e: # Memory growth must be set before GPUs have been initialized
        print(e)


EPOCH = 100
BATCH_SIZE = 32

# tfrecord decode
def parse_image(record):
    features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                'image/encoded': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_record = tf.io.parse_single_example(record, features)
    label= tf.cast(parsed_record['image/source_id'], tf.int32)

#    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.decode_jpeg(parsed_record['image/encoded'], channels=3)
    return image, label

# train dataloader (augmentation)
def get_dataset_train(path, batch_size=BATCH_SIZE):
    dataset = tf.data.Dataset.list_files(path, seed=42)    # seed: shuffle

    dataset = tf.data.TFRecordDataset(filenames=dataset, compression_type="GZIP")
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset = < ParallelMapDataset shapes: ((None, None, 3), ()), types: (tf.uint8, tf.int32) >
    # Data Agumentation
    dataset = dataset.map(lambda image, label : (tf.image.random_flip_left_right(image), label), num_parallel_calls=tf.data.experimental.AUTOTUNE)             # flip
    dataset = dataset.map(lambda image, label: (tf.image.random_crop(image, size=[224, 224, 3]), label), num_parallel_calls=tf.data.experimental.AUTOTUNE)     # crop
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=10)

    return dataset

# val or test dataloader (no augmentation)
def get_dataset_val(path, batch_size=BATCH_SIZE):
    dataset = tf.data.Dataset.list_files(path, seed=42)
    dataset = tf.data.TFRecordDataset(filenames=dataset, compression_type="GZIP")
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda image, label: (tf.image.resize_with_pad(image, 224, 224), label), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def create_vgg_16():

    input_data = Input(shape=(224, 224, 3))

    x = Conv2D(filters=64, kernel_size=(3,3), padding="SAME", activation='relu')(input_data)
    x = Conv2D(filters=64, kernel_size=(3,3), padding="SAME", activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2, 2))(x)

    x = Conv2D(filters=128, kernel_size=(3,3), padding="SAME", activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3,3),  padding="SAME", activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2, 2), padding="SAME")(x)

    x = Conv2D(filters=256, kernel_size=(3,3), padding="SAME", activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(3,3), padding="SAME", activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(3,3), padding="SAME", activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2, 2))(x)

    x = Conv2D(filters=512, kernel_size=(3,3), padding="SAME", activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="SAME", activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="SAME", activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2, 2))(x)

    x = Conv2D(filters=512, kernel_size=(3,3), padding="SAME", activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="SAME", activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="SAME", activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(4096, activation="relu")(x)  ## FC
    x = Dropout(0.5)(x)
    x = Dense(4096, activation="relu")(x)  ## FC
    x = Dropout(0.5)(x)

    output = Dense(1000, activation="softmax")(x)

    model = Model(inputs=input_data, outputs=output, name='vgg-16')
    model.summary()
    return model

def main():

    train_url = "/home/ivpl-d14/PycharmProjects/pythonProject/model_implementation/Model-Implementation/tfrecords/tf_train/train2.tfrecord"
    val_url = "/home/ivpl-d14/PycharmProjects/pythonProject/model_implementation/Model-Implementation/tfrecords/tf_train/val2.tfrecord"

    train_dataset = get_dataset_train(train_url)
    val_dataset = get_dataset_val(val_url)

    model = create_vgg_16()
    #model.summary()



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

    model.fit(train_dataset,
              validation_data=val_dataset,
              validation_steps=validation_steps,
              epochs=EPOCH,
              batch_size=BATCH_SIZE,
              steps_per_epoch=steps_per_epoch,
              callbacks=callbacks)
    ##initial_epoch=100,

    # 모델 저장
    model.save('my_vgg_16.h5')



if __name__ == '__main__':
    main()
