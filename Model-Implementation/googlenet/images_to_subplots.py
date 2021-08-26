import glob
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import cv2

fig = plt.figure()  # rows*cols 행렬의 i번째 subplot 생성
rows = 8
cols = 8
i = 1

xlabels = []
dir = "/home/ivpl-d14/PycharmProjects/imagenet/imagenet/test/zucchini/*.JPEG"

for filename in glob.glob(dir):
    img = cv2.imread(filename)
    ax = fig.add_subplot(rows, cols, i)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_xlabel(str(i))
    ax.set_xticks([]), ax.set_yticks([])
    i += 1

plt.tight_layout()
plt.show()
