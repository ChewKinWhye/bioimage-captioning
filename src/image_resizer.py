from PIL import Image
import os

base_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
image_folder = os.path.join(base_path, "data", "indiana-university","images", "images_normalized")

c = 0
for f in os.listdir(image_folder):
    basewidth = 300
    fp = os.path.join(image_folder,f)
    img = Image.open(fp)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img.save(fp)
    c += 1

print("RESIZED images:", c)