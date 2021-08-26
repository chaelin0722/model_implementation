import tensorflow as tf
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

#데이터셋 설정
train_datagen = ImageDataGenerator(rescale=1./255,width_shift_range=0.2,height_shift_range =0.2,
    zoom_range=0.2, horizontal_flip =True, vertical_flip = True)

train_generator = train_datagen.flow_from_directory(
    './catdog/training_set/training_set/',
    target_size=(227, 227),
    batch_size=16,
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255,width_shift_range=0.2,height_shift_range =0.2,
zoom_range=0.2, horizontal_flip =True, vertical_flip = True)

test_generator = test_datagen.flow_from_directory(
    './catdog/test_set/test_set/',
    target_size=(227, 227),
    batch_size=16,
    class_mode='categorical')



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
    tf.keras.layers.Dense(2, activation='softmax')
])

sgd = SGD(lr=0.01,decay=5e-4, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer='sgd',metrics=['accuracy'])

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
callback_list = [tensorboard_callback]

model.summary()

# start training
model.fit_generator(train_generator, steps_per_epoch=50, epochs=10)


#모델 저장하기
model.save('my_Alexnet.h5')

#모델 평가하기
print("-------------Evaluate-----------------")
scores = model.evaluate_generator(test_generator,steps=1)
print("%s : %.2f%%" %(model.metrics_names[1],scores[1]*100))
