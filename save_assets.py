from numpy import asarray
from numpy import savez_compressed

import numpy as np
import os
import torchvision

from PIL import Image


dir_data = "assets"
img_shape = (64, 64, 3)
nm_imgs = np.sort(os.listdir(dir_data))

X_train = []
for file in nm_imgs:
    try:
        img = Image.open(dir_data+'/'+file)
        img = img.convert('RGB')
        img = img.resize((64,64))
        img = np.asarray(img)/255
        X_train.append(img)
    except:
        print("something went wrong")
 
X_train = np.array(X_train)
print(X_train.shape)
savez_compressed('spaceships.npz', X_train)