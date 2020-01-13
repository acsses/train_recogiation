from PIL import Image
import os
import sys

path = sys.argv

folder = os.listdir(path[1])
print(folder)

for filename in folder:
    full_name = path[1] + "/" + filename
    im = Image.open(full_name)
    cropped = im.crop(im.getbbox())
    width,height=8,8
    img = cropped.resize((width,height))
    img.save(full_name)