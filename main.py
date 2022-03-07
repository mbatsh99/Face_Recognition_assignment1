import pandas as pd
import numpy as np
import sklearn as sklearn
import seaborn as sb
from PIL import Image
from matplotlib import image
from matplotlib import pyplot
import glob

path = '/Users/batsh/Desktop/ORL dataset/'
def get_all_images(path):
    image_list = []
    for x in range(1,41): #loop on every subject folder
        for filename in glob.glob(path + 's' + str(x) + '/*.pgm'): #get all images inside sX folder
            Im = Image.open(filename)
            image_list.append(Im)
    return image_list


images = get_all_images(path)
print(len(images)) #400

#convert images list to a numpy array list
images_vector = []
for img in images:
    images_vector.append(np.array(img).ravel())

print(len(images_vector)) #400
print(images_vector[0].size) #10304
print(images_vector[0].shape) #10304

df = pd.DataFrame(images_vector)
y_labels = []
for i in range(1,41):
    for j in range(1,11):
        y_labels.append(i)
df['label'] = pd.Series(y_labels)
print(df.head())
print(df.tail())


