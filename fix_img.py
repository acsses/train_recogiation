from PIL import Image
import os
import sys

path = sys.argv

folder = os.listdir(path[1])
print(folder)

for filename in folder:
    full_name = path[1] + "/" + filename
    im = Image.open(full_name)
    fix_img = im.convert('L')
    cropped = fix_img.crop(fix_img.getbbox())
    width,height=64,64
    img = cropped.resize((width,height),Image.ANTIALIAS)
    img.save(full_name)