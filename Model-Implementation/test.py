import os


#charr = '898'

#toint = int(charr)
#print(i+toint)
#img_path = "/home/ivpl-d14/PycharmProjects/pythonProject/model_implementation/Model-Implementation/tfrecords/tf_train/train.tfrecord"

#filename = os.path.join(str(i), os.path.basename(img_path))

#a = 'python'

#print(a[:-1])

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

'''
BATCH_SIZE = 32

import tensorflow as tf


# tfrecord decode
def parse_image(record):
    features = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_record = tf.io.parse_single_example(record, features)
    image = tf.io.decode_jpeg(parsed_record['image_raw'], channels=3)
    label= tf.cast(parsed_record['label'], tf.int32)

    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label

# train dataloader (augmentation)
def get_dataset_train(path, batch_size=BATCH_SIZE):
    record_files = tf.data.Dataset.list_files(path, seed=42)    # seed: shuffle
    dataset = tf.data.TFRecordDataset(filenames=record_files, compression_type="GZIP")
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

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
    record_files = tf.data.Dataset.list_files(path, seed=42)
    dataset = tf.data.TFRecordDataset(filenames=record_files, compression_type="GZIP")
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda image, label: (tf.image.resize_with_pad(image, 224, 224), label), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset=dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset=dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

# tfrecord data load
train_dataset= get_dataset_train("/home/hjpark/pycharmProject/Alexnet/imagenet_prep/tf_records/train/*.tfrecord")
val_dataset= get_dataset_val("/home/hjpark/pycharmProject/Alexnet/imagenet_prep/tf_records/val/*.tfrecord")
'''

from glob import glob
import os
import random
import tensorflow as tf


def serialize_example(image, label):
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size


def make_tfrecords(path, record_file):
    print('start')

    file_index_count = 0
    classes = os.listdir(path)
    classes.sort()
    files_list = glob(path + '/*/*')
    random.shuffle(files_list)
    writer = tf.io.TFRecordWriter(record_file.format(file_index_count))
    print(f'Total image files: {len(files_list)}')
    print(f'TFRecord number: {file_index_count}')

    for filename in files_list:
        image_string = open(filename, 'rb').read()
        category = filename.split('/')[-2]
        label = classes.index(category)
        tf_example = serialize_example(image_string, label)
        writer.write(tf_example)
        # print(f'class:{label}__{filename}')

        size = get_file_size(record_file.format(file_index_count))
        mg_size = round(size / (1024 * 1024), 3)
        # print('File size: ' + str(mg_size) + ' Megabytes')

        if mg_size > 100:
            file_index_count += 1
            writer = tf.io.TFRecordWriter(record_file.format(file_index_count))
            print(f'TFRecord number: {file_index_count}, {files_list.index(filename)}')

    print('done')


def main():

    dataset_path = "/home/ivpl-d14/PycharmProjects/imagenet/imagenet/val"
    output_path = "/home/ivpl-d14/PycharmProjects/pythonProject/model_implementation/Model-Implementation/tfrecords/tf_train/val_ji.tfrecord"
    make_tfrecords(dataset_path, output_path)





if __name__ == '__main__':
    main()
