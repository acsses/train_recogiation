from PIL import Image
import numpy as np
import os

i = 0
data_array = []
target_array = np.array([])
main_folder = os.listdir("./data/")
for sub_folder in main_folder:
    data_list = os.listdir("./data/" + sub_folder)
    for data in data_list:
        img = Image.open("./data/" + sub_folder + "/" + data)
        fix_img = img.convert('L')
        data_grey = np.asarray(fix_img)
        target_array = np.append(target_array,sub_folder)
        data_array.append(data_grey)
        i = i +1
data = np.array(data_array)
np.save("./data",data)
np.save("./target",target_array)
print(data[None].shape)