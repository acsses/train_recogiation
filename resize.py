from PIL import Image
import os
folder = os.listdir("./test/data/321/")
print(folder)
for filename in folder:
    full_name = "./test/data/321/" + filename
    im = Image.open(full_name)
    width,height=8,8
    img = im.resize((width,height),Image.ANTIALIAS)
    img.save(full_name)