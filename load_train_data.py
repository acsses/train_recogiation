from PIL import Image
import numpy as np

img = Image.open("./スクリーンショット 2020-01-09 7.30.05.png")
fix_img = img.convert('L')
dat_grey = np.asarray(fix_img)
print(dat_grey)
