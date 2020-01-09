from PIL import Image
import numpy as np
import os

data_array = np.asarray([])

main_folder = os.listdir("./")
for sub_folder in main_folder:
    data_ist = os.listdir(sub_folder)
    for data in sub_folder:
        img = Image.open(data)
        fix_img = img.convert('L')
        dat_grey = np.asarray(fix_img)
