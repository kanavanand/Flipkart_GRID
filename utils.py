import pandas as pd
#from google.colab import  files
import random
import cv2
import numpy as np
from scipy.ndimage import affine_transform
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
from keras import backend as K
from keras.preprocessing.image import array_to_img
from numpy.linalg import inv as mat_inv
from keras.utils import Sequence
import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
import pandas as pd
import numpy as np

from keras.engine.topology import Input
from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Model
#from __future__ import print_function, division
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage import io, transform, img_as_float, measure
from PIL import Image as pil_image
from PIL.ImageDraw import Draw
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from keras.applications.xception import Xception, preprocess_input
#from keras.layers import*
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.losses import mean_squared_error
from keras.utils import Sequence
from keras import backend as K

import cv2

from IPython.display import clear_output
from keras.engine.topology import Input
from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Model
#from __future__ import print_function, division

from sklearn.model_selection import train_test_split
from skimage import io, transform, img_as_float, measure

from PIL import Image as pil_image
from PIL.ImageDraw import Draw

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from keras.preprocessing.image import ImageDataGenerator
# from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model

#plt.ion()   # interactive mode
import numpy as np
import pandas as pd
import gc
import keras
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy
from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import  ModelCheckpoint
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from tqdm import tqdm_notebook
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
from keras import backend as K
from keras.layers import LeakyReLU
from keras.layers import ZeroPadding2D
from keras.losses import binary_crossentropy
import keras.callbacks as callbacks
from keras.callbacks import Callback
from keras.applications.xception import Xception
from keras.layers import multiply
from keras import optimizers
from keras.legacy import interfaces
from keras.utils.generic_utils import get_custom_objects
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.regularizers import l2
from keras.layers.core import Dense, Lambda
from keras.layers.merge import concatenate, add
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from keras.optimizers import SGD

#plt.ion()   # interactive mode
from PIL import Image as pil_image
from PIL.ImageDraw import Draw
from os.path import isfile
from config import*

# This function will check whether the file is present in that folder,if yes-> it will return exact path else->None
def expand_path(p):
    if isfile(image_direc + p): return image_direc + p
    return p

# Read a raw image from path.

def read_raw_image(p):
    return pil_image.open(expand_path(p))



# Read an image as numpy array and resize it 
def read_array(p):
    img = img_to_array(read_raw_image(p))
    img =  cv2.resize(img,None,fx=224/640, fy=224/480)
    return img

# Read an image for validation and DNORM it.
def read_for_validation(p):
    x  = read_array(p)
    x -= np.mean(x, keepdims=True)
    x /= np.std(x, keepdims=True) + K.epsilon()
    return x

# Read an image for training and DNORM it.
def read_for_training(p):
    x  = read_array(p)
    x -= np.mean(x, keepdims=True)
    x /= np.std(x, keepdims=True) + K.epsilon()
    return x 


def get_mask_seg(img,bb_box_i):

    img_mask = np.zeros_like(img[:,:,0])
    img_mask[np.int32(bb_box_i[1]):np.int(bb_box_i[3]),np.int(bb_box_i[0]):np.int(bb_box_i[2])]= 1.
    img_mask = np.reshape(img_mask,(np.shape(img_mask)[0],np.shape(img_mask)[1],1))
    return img_mask

import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
from keras import backend as K
from keras.preprocessing.image import array_to_img
from numpy.linalg import inv as mat_inv
import numpy as np
def show_object(imgs, per_row=5):
    n         = len(imgs)
    rows      = (n + per_row - 1)//per_row
    cols      = min(per_row, n)
    fig, axes = plt.subplots(rows,cols, figsize=(12//per_row*cols,10//per_row*rows))
    for ax in axes.flatten(): ax.axis('off')
    for i,(img,ax) in enumerate(zip(imgs, axes.flatten())): ax.imshow(img.convert('RGB'))



print("All the packages are loaded")
print("building model......")