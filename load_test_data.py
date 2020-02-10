from PIL import Image
import numpy as np
import os

i = 0
data_array = []
target_array = np.array([])

img = Image.open("./test_data/test.png")
data_grey = np.asarray(img)
print(data_grey.shape)
np.save("./test_data",data_grey)

