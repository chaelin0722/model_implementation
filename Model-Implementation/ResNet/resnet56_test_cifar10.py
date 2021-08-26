
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np

def main():

    # Load Cifar-10 data-set
    (train_im, train_lab), (test_im, test_lab) = tf.keras.datasets.cifar10.load_data()

    #### Normalize the images to pixel values (0, 1)
    train_im, test_im = train_im / 255.0, test_im / 255.0

    classes =['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


    ### One hot encoding for labels
    test_lab_categorical = tf.keras.utils.to_categorical(test_lab, num_classes=10, dtype='uint8')
    train_lab_categorical = tf.keras.utils.to_categorical(train_lab, num_classes=10, dtype='uint8')

    BATCH_SIZE = 128

    model = keras.models.load_model('my_resnet_56.h5')

    train_lab = train_lab.reshape(-1, )


    pred = model.predict(test_im)

    result = [np.argmax(value) for value in pred]   # 예측 값중 가장 높은 클래스 반환
    '''
    def plot_sample(X):
        plt.figure(figsize=(15, 2))
        plt.imshow(X)
        plt.xlabel(result)


    for i in range(3):
        plot_sample(test_im)
   '''
    for i in range(5):
        plt.figure(figsize=(15, 4))
        plt.imshow(test_im[i])
        plt.title(classes[result[i]])

        print('data category result : ', result[i])
    plt.show()

'''
    results = model.evaluate(test_im, test_lab_categorical, batch_size=BATCH_SIZE)

    print("Test loss: ", results[0])
    print("Test accuracy: ", results[1])
'''

if __name__ == '__main__':
    main()