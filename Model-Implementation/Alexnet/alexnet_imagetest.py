
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import load_model

model = load_model('Alexnet.h5')
image_url ='./catdog/test_set/test_set/'

test_datagen = ImageDataGenerator(rescale=1./255,width_shift_range=0.2,height_shift_range =0.2,
zoom_range=0.2, horizontal_flip =True, vertical_flip = True)

test_generator = test_datagen.flow_from_directory(
    image_url,
    target_size=(227, 227),
    batch_size=128,
    class_mode='categorical')
#####
class_names = ['cat', 'dog']

x_valid, label_batch = next(iter(test_generator))


prediction_values = np.argmax(model.predict(x_valid), axis=-1)

# set up the figure
fig = plt.figure(figsize=(10, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# plot the images: each image is 227x227 pixels
for i in range(8):
    ax = fig.add_subplot(2, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(x_valid[i, :], cmap=plt.cm.gray_r, interpolation='nearest')

    if prediction_values[i] == np.argmax(label_batch[i]):
        # label the image with the blue text
        ax.text(3, 17, class_names[prediction_values[i]], color='blue', fontsize=14)
    else:
        # label the image with the red text
        ax.text(3, 17, class_names[prediction_values[i]], color='red', fontsize=14)