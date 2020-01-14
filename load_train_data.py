from PIL import Image
import numpy as np
import os


def load_func():
    data_array = np.asarray([])
    target_arrry = np.array([])
    main_folder = os.listdir("./")
    for sub_folder in main_folder:
        data_list = os.listdir(sub_folder)
        for data in data_list:
            img = Image.open(data)
            fix_img = img.convert('L')
            data_grey = np.asarray(fix_img)
            target_arrry = np.append(target_arrry,sub_folder)
            data_array = np.append(data_array,data_grey,axis=0)
    return data_array,np.array([target_arrry]).astype(np.int32)
