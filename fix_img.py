from PIL import Image
import os
folder = os.listdir("./Type223Seris1000/")
print(folder)
for filename in folder:
    full_name = "./Type223Seris1000/" + filename
    im = Image.open(full_name)
    cropped = im.crop(im.getbbox())
    cropped.save(full_name)