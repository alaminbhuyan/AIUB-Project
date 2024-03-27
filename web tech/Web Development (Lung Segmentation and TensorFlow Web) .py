#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy.random import seed
seed(101)

import pandas as pd
import numpy as np


import os
import cv2

import imageio
import skimage
import skimage.io
import skimage.transform

from skimage.io import imread, imshow
from skimage.transform import resize


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


os.listdir("G:\Web")


# In[6]:


NUM_TEST_IMAGES = 100

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_CHANNELS = 3

BATCH_SIZE = 100


# In[7]:


# Dataset: pulmonary-chest-xray-abnormalities
# Here only the Montgomery images have masks. The Shenzhen don't have masks.

os.listdir('G:\Web\Dataset1')


# In[8]:


# Dataset: shcxr-lung-mask dataset
# These are the masks for the Shenzhen images.

os.listdir('G:\Web\mask')

The label is part of the file name.

Example: CHNCXR_0470__1.png

0 = Normal (No TB)
1 = TB

Each of the two datasets has a text file containing meta-data.
# In[10]:


shen_image_list = \
os.listdir('G:\\Web\\Dataset1\\ChinaSet_AllFiles\\ChinaSet_AllFiles\\CXR_png')
shen_mask_list = os.listdir('G:\\Web\\mask\\mask')

mont_image_list = \
os.listdir('G:\\Web\\Dataset1\\Montgomery\\MontgomerySet\\CXR_png')
mont_left_mask_list = \
os.listdir('G:\\Web\Dataset1\\Montgomery\\MontgomerySet\\ManualMask\\leftMask')
mont_right_mask_list = \
os.listdir('G:\\Web\\Dataset1\\Montgomery\\MontgomerySet\\ManualMask\\rightMask')



def find_non_images(image_list):
  
  """
  Checks a list and returns a list of items 
  that are not png images.
  
  """
  
  non_image_list = []
  
  for fname in image_list:
    # split on the full stop
    fname_list = fname.split('.')

    # select the extension
    extension = fname_list[1]

    if extension != 'png':
      non_image_list.append(fname)

  return non_image_list


# In[11]:


# Non images in Shenzhen folder
non_images = find_non_images(shen_image_list)

non_images


# In[12]:


# Non images in Shenzhen mask folder
non_images = find_non_images(shen_mask_list)

non_images


# In[13]:


# Non images in Montgomery image folder
non_images = find_non_images(mont_image_list)

non_images


# In[14]:


# Non images in Montgomery right mask folder
non_images = find_non_images(mont_right_mask_list)

non_images


# In[15]:


# Mongomery images
# 138 (excl. Thumbs.db)
print(len(os.listdir('G:\\Web\\Dataset1\\Montgomery\\MontgomerySet\\CXR_png')))

# Mongomery left masks
# 138 (excl. Thumbs.db)
print(len(os.listdir('G:\\Web\Dataset1\\Montgomery\\MontgomerySet\\ManualMask\\leftMask')))

# Mongomery right masks
# 138
print(len(os.listdir('G:\\Web\\Dataset1\\Montgomery\\MontgomerySet\\ManualMask\\rightMask')))



# Shenzhen images
# 662 (excl. Thumbs.db)
print(len(os.listdir('G:\\Web\\Dataset1\\ChinaSet_AllFiles\\ChinaSet_AllFiles\\CXR_png')))

# Shenzhen masks
# 566
print(len(os.listdir('G:\\Web\\mask\\mask')))


# In[16]:


shen_image_list = os.listdir('G:\\Web\\Dataset1\\ChinaSet_AllFiles\\ChinaSet_AllFiles\\CXR_png')

mont_image_list = os.listdir('G:\\Web\\Dataset1\\Montgomery\\MontgomerySet\\CXR_png')


# In[17]:


# put the images into dataframes
df_shen = pd.DataFrame(shen_image_list, columns=['image_id'])
df_mont = pd.DataFrame(mont_image_list, columns=['image_id'])

# remove the 'Thunbs.db' line
df_shen = df_shen[df_shen['image_id'] != 'Thumbs.db']
df_mont = df_mont[df_mont['image_id'] != 'Thumbs.db']

# Reset the index or this will cause an error later
df_shen.reset_index(inplace=True, drop=True)
df_mont.reset_index(inplace=True, drop=True)

print(df_shen.shape)
print(df_mont.shape)


# In[18]:


df_shen.head()


# In[19]:


df_mont.head()


# In[21]:


# Put the Shenzhen masks into a dataframe

shen_mask_list = os.listdir('G:\\Web\\mask\\mask')

df_shen_masks = pd.DataFrame(shen_mask_list, columns=['mask_id'])

# create a new column with the image_id that corresponds to each mask

# example mask_id: CHNCXR_0001_0_mask.png

def create_image_id(x):
  
  # split on '_mask'
  fname_list = x.split('_mask')
  image_id = fname_list[0] + fname_list[1]
  
  return image_id
  
# create a new column
df_shen_masks['image_id'] = df_shen_masks['mask_id'].apply(create_image_id)

df_shen_masks.head()


# In[22]:


df_shen = pd.merge(df_shen, df_shen_masks, on='image_id')

df_shen.head()


# In[23]:


df_shen.shape


# In[25]:


# Function to select the 4th index from the end of the string (file name)
# CHNCXR_0470_1.png --> 1 is the label, meaning TB is present.

def extract_target(x):
    target = int(x[-5])
    if target == 0:
        return 'Normal'
    if target == 1:
        return 'Tuberculosis'


# In[26]:


# Assign the target labels

df_shen['target'] = df_shen['image_id'].apply(extract_target)

df_mont['target'] = df_mont['image_id'].apply(extract_target)


# In[27]:


# Shenzen Dataset

df_shen['target'].value_counts()


# In[28]:


# Montgomery Dataset

df_mont['target'].value_counts()


# In[29]:


def draw_category_images(col_name,figure_cols, df, IMAGE_PATH):
    
    """
    Give a column in a dataframe,
    this function takes a sample of each class and displays that
    sample on one row. The sample size is the same as figure_cols which
    is the number of columns in the figure.
    Because this function takes a random sample, each time the function is run it
    displays different images.
    """
    

    categories = (df.groupby([col_name])[col_name].nunique()).index
    f, ax = plt.subplots(nrows=len(categories),ncols=figure_cols, 
                         figsize=(4*figure_cols,4*len(categories))) # adjust size here
    # draw a number of images for each location
    for i, cat in enumerate(categories):
        sample = df[df[col_name]==cat].sample(figure_cols) # figure_cols is also the sample size
        for j in range(0,figure_cols):
            file=IMAGE_PATH + sample.iloc[j]['image_id']
            im=imageio.imread(file)
            ax[i, j].imshow(im, resample=True, cmap='gray')
            ax[i, j].set_title(cat, fontsize=14)  
    plt.tight_layout()
    plt.show()


# In[32]:


# Shenzen Dataset

IMAGE_PATH = 'G:\\Web\\Dataset1\\ChinaSet_AllFiles\\ChinaSet_AllFiles\\CXR_png\\'

draw_category_images('target',4, df_shen, IMAGE_PATH)


# In[33]:


# Montgomery Dataset

IMAGE_PATH = 'G:\\Web\\Dataset1\\Montgomery\\MontgomerySet\\CXR_png\\'

draw_category_images('target',4, df_mont, IMAGE_PATH)


# In[35]:


def read_image_sizes(file_name):
    image = cv2.imread(IMAGE_PATH + file_name)
    max_pixel_val = image.max()
    min_pixel_val = image.min()
    
    # image.shape[2] represents the number of channels: (height, width, num_channels).
    # Here we are saying: If the shape does not have a value for num_channels (height, width)
    # then assign 1 to the number of channels.
    if len(image.shape) > 2: # i.e. more than two numbers in the tuple
        output = [image.shape[0], image.shape[1], image.shape[2], max_pixel_val, min_pixel_val]
    else:
        output = [image.shape[0], image.shape[1], 1, max_pixel_val, min_pixel_val]
    return output


# In[36]:


IMAGE_PATH = 'G:\\Web\\Dataset1\\ChinaSet_AllFiles\\ChinaSet_AllFiles\\CXR_png\\'

m = np.stack(df_shen['image_id'].apply(read_image_sizes))
df = pd.DataFrame(m,columns=['w','h','c','max_pixel_val','min_pixel_val'])
df_shen = pd.concat([df_shen,df],axis=1, sort=False)

df_shen.head()


# In[37]:


IMAGE_PATH = 'G:\\Web\\Dataset1\\Montgomery\\MontgomerySet\\CXR_png\\'

m = np.stack(df_mont['image_id'].apply(read_image_sizes))
df = pd.DataFrame(m,columns=['w','h','c','max_pixel_val','min_pixel_val'])
df_mont = pd.concat([df_mont,df],axis=1, sort=False)

df_mont.head()


# In[38]:


df_shen['c'].value_counts()


# In[39]:


df_mont['c'].value_counts()

We see that all images have 3 channels. The images have a pixel value range between 0 and 255.
# # Display one Montgomery image and mask

# In the Mongomery dataset there are separate masks for the left lung and right lung. We will combine these two masks into one by simply adding the matrices. 
# Each mask has the same file name as it's corresponding image.

# In[40]:


# print a Montgomery image and mask

# image
index = 2
fname = df_mont.loc[index, 'image_id']
path = 'G:\\Web\\Dataset1\\Montgomery\\MontgomerySet\\CXR_png\\' + fname
# read the image as a matrix
image = plt.imread(path)

plt.imshow(image, cmap='gray')


# In[41]:


fname = df_mont.loc[index, 'image_id']


# In[44]:


# left mask
path = 'G:\\Web\\Dataset1\\Montgomery\\MontgomerySet\\ManualMask\\leftMask\\' + fname
left_mask = plt.imread(path)

# right mask
path = 'G:\\Web\\Dataset1\\Montgomery\\MontgomerySet\\ManualMask\\rightMask\\' + fname
right_mask = plt.imread(path)

# combine both masks
mask = left_mask + right_mask

plt.imshow(mask)


# In[45]:


# display the Montgomery image and mask

plt.imshow(image, cmap='gray')
plt.imshow(mask, cmap='Blues', alpha=0.3)


# In[46]:


index = 3
fname = df_shen.loc[index, 'image_id']
path = 'G:\\Web\\Dataset1\\ChinaSet_AllFiles\\ChinaSet_AllFiles\\CXR_png\\' + fname

# read the image as a matrix
image = plt.imread(path)

plt.imshow(image, cmap='gray')


# In[47]:


fname = df_shen.loc[index, 'image_id']

mask_name = fname.split('.')
mask_name = mask_name[0] + '_mask.png'


# left mask
path = 'G:\\Web\\mask\\' + mask_name
mask = plt.imread(path)


plt.imshow(mask)


# In[48]:


# display the Shenzhen image and mask

plt.imshow(image, cmap='gray')
plt.imshow(mask, cmap='Blues', alpha=0.3)


# # Create a dataframe containing all images

# In[49]:


### Combine the two dataframes and shuffle

df_data = pd.concat([df_shen, df_mont], axis=0).reset_index(drop=True)

df_data = shuffle(df_data)


df_data.shape


# In[50]:


# Create a new column called 'labels' that maps the classes to binary values.
df_data['labels'] = df_data['target'].map({'Normal':0, 'Tuberculosis':1})


# In[51]:


df_data.head()


# # Create a holdout test set

# In[52]:


# create a test set
df_test = df_data.sample(NUM_TEST_IMAGES, random_state=101)

# Reset the index.
df_test = df_test.reset_index(drop=True)

# create a list of test images
test_images_list = list(df_test['image_id'])


# Select only rows that are not part of the test set.
# Note the use of ~ to execute 'not in'.
df_data = df_data[~df_data['image_id'].isin(test_images_list)]

print(df_data.shape)
print(df_test.shape)


# # Train Test Split

# In[55]:


# train_test_split

# We will stratify by target (TB or Normal)

y = df_data['labels']

df_train, df_val = train_test_split(df_data, test_size=0.4, random_state=101, stratify=y)

print(df_train.shape)
print(df_val.shape)


# In[56]:


df_train['target'].value_counts()


# In[57]:


df_val['target'].value_counts()


# # Save the dataframes as compressed csv files

# In[58]:


df_data.to_csv('df_data.csv.gz', compression='gzip', index=False)

df_train.to_csv('df_train.csv.gz', compression='gzip', index=False)
df_val.to_csv('df_val.csv.gz', compression='gzip', index=False)

df_test.to_csv('df_test.csv.gz', compression='gzip', index=False)


# In[60]:


# Create a new directory
image_dir = 'image_dir'
os.mkdir(image_dir)


# In[66]:


get_ipython().run_cell_magic('time', '', "# Get a list of train and val images\nshen_image_list = list(df_shen['image_id'])\nmont_image_list = list(df_mont['image_id'])\n")


# In[67]:


# Transfer the Shenzhen images

for image_id in shen_image_list:   
    
    fname = image_id
    
    path = 'G:\\Web\\mask\\ChinaSet_AllFiles\\ChinaSet_AllFiles\\CXR_png\\' + fname
    # read the image
    image = cv2.imread(path)
    
    # convert to from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # resize the image
    image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    
    # save the image
    path = 'image_dir/' + fname
    cv2.imwrite(path, image)


# In[68]:


# Transfer the Montgomery images

for image_id in mont_image_list: 
  
    fname = image_id
    
    path = 'G:\\Web\\mask\\Montgomery\\MontgomerySet\\CXR_png\\' + fname
    # read the image
    image = cv2.imread(path)
    
    # convert to from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # resize the image
    image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    
    # save the image
    path = 'image_dir/' + fname
    cv2.imwrite(path, image)


# In[70]:


get_ipython().run_cell_magic('time', '', "\n# Get a list of train and val images\nshen_mask_list = list(df_shen['mask_id'])\nmont_mask_list = list(df_mont['image_id'])\n")


# In[ ]:





# In[71]:


# Transfer the Shenzhen masks
# These masks have file names that are not the same as the images

for image in shen_mask_list:
    
    
    fname = image
    
    # change the mask file name to be the same as the image_id
    fname_list = fname.split('_mask')
    new_fname = fname_list[0] + fname_list[1]
    
    path = 'G:\\Web\\mask\\' + fname
    # read the image
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    # resize the mask
    mask = cv2.resize(mask, (IMAGE_HEIGHT, IMAGE_WIDTH))
    
    # save the mask
    path = 'mask_dir/' + new_fname
    cv2.imwrite(path, mask)


# In[72]:


# Transfer the Montgomery masks

for image in mont_mask_list:
    
    
    fname = image
    
    
    # left mask
    path = 'G:\\Web\\mask\\leftMask\\' + fname
    left_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # right mask
    path = 'G:\\Web\\mask\\rightMask\\' + fname
    right_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # combine left and right masks
    mask = left_mask + right_mask
    
    # resize the mask
    mask = cv2.resize(mask, (IMAGE_HEIGHT, IMAGE_WIDTH))
    
    
    # save the combined mask
    path = 'mask_dir/' + fname
    cv2.imwrite(path, mask)


# # [ 1 ] Train Generator

# In[73]:


def train_generator(batch_size=10):
    
    while True:
        
        # load the data in chunks (batches)
        for df in pd.read_csv('df_train.csv.gz', chunksize=batch_size):
            
            # get the list of images
            image_id_list = list(df['image_id'])
            mask_id_list = list(df['image_id'])
            
            # Create empty X matrix - 3 channels
            X_train = np.zeros((len(df), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8)
            
            # create empty Y matrix - 1 channel
            Y_train = np.zeros((len(df), IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.bool)

        
            
            # Create X_train
            #================
            
            for i, image_id in enumerate(image_id_list):
                

                # set the path to the image
                path = 'image_dir/' + image_id

                # read the image
                image = cv2.imread(path)
                
                # convert to from BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # resize the image
                #image = resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH), mode='constant', preserve_range=True)
                
                # insert the image into X_train
                X_train[i] = image
            
            
            # Create Y_train
            # ===============
                
            for j, mask_id in enumerate(mask_id_list):

                # set the path to the mask
                path = 'mask_dir/' + mask_id

                # read the mask
                mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                
                # expand dims from (800,600) to (800,600,1)
                mask = np.expand_dims(mask, axis=-1)
                
                # resize the mask
                #mask = resize(mask, (IMAGE_HEIGHT, IMAGE_WIDTH), mode='constant', preserve_range=True)
                
                
                # insert the image into Y_train
                Y_train[j] = mask
                
                
            # Normalize the images
            X_train = X_train/255

            yield X_train, Y_train


# In[74]:


# Test the generator

# initialize
train_gen = train_generator(batch_size=10)

# run the generator
X_train, Y_train = next(train_gen)

print(X_train.shape)
print(Y_train.shape)


# In[75]:


# print the first image in X_train

img = X_train[7,:,:,:]
plt.imshow(img)


# In[76]:


# print the first mask in Y_train

msk = Y_train[7,:,:,0]
plt.imshow(msk)


# In[77]:


plt.imshow(img, cmap='gray')
plt.imshow(msk, cmap='Blues', alpha=0.3)


# # [ 2 ] Val Generator

# In[78]:


def val_generator(batch_size=10):
    
    while True:
        
        # load the data in chunks (batches)
        for df in pd.read_csv('df_val.csv.gz', chunksize=batch_size):
            
            # get the list of images
            image_id_list = list(df['image_id'])
            mask_id_list = list(df['image_id'])
            
            # Create empty X matrix - 3 channels
            X_val = np.zeros((len(df), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8)
            
            # create empty Y matrix - 1 channel
            Y_val = np.zeros((len(df), IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.bool)

        
            
            # Create X_val
            #================
            
            for i, image_id in enumerate(image_id_list):
                

                # set the path to the image
                path = 'image_dir/' + image_id

                # read the image
                image = cv2.imread(path)
                
                # convert to from BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # resize the image
                #image = resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH), mode='constant', preserve_range=True)
                
                # insert the image into X_train
                X_val[i] = image
            
            
            # Create Y_val
            # ===============
                
            for j, mask_id in enumerate(mask_id_list):

                # set the path to the mask
                path = 'mask_dir/' + mask_id

                # read the mask
                mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                
                # expand dims from (800,600) to (800,600,1)
                mask = np.expand_dims(mask, axis=-1)
                
                # resize the mask
                #mask = resize(mask, (IMAGE_HEIGHT, IMAGE_WIDTH), mode='constant', preserve_range=True)
                
                
                # insert the image into Y_train
                Y_val[j] = mask
                
            
            # Normalize the images
            X_val = X_val/255
            
            yield X_val, Y_val


# In[79]:


# Test the generator

# initialize
val_gen = val_generator(batch_size=10)

# run the generator
X_val, Y_val = next(val_gen)

print(X_val.shape)
print(Y_val.shape)


# In[80]:


# print the image from X_val

img = X_val[7,:,:,:]
plt.imshow(img)


# In[81]:


# print the mask from Y_val

msk = Y_val[7,:,:,0]
plt.imshow(msk)


# In[82]:


# Combine the mask and the image

plt.imshow(img, cmap='gray')
plt.imshow(msk, cmap='Blues', alpha=0.3)


# # [ 3 ] Test Generator

# In[83]:


def test_generator(batch_size=1):
    
    while True:
        
        # load the data in chunks (batches)
        for df in pd.read_csv('df_test.csv.gz', chunksize=batch_size):
            
            # get the list of images
            image_id_list = list(df['image_id'])
            mask_id_list = list(df['image_id'])
            
            # Create empty X matrix - 3 channels
            X_test = np.zeros((len(df), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8)
            
            # create empty Y matrix - 1 channel
            Y_test = np.zeros((len(df), IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.bool)
            


            
            # Create X_test
            #================
            
            for i, image_id in enumerate(image_id_list):
                

                # set the path to the image
                path = 'image_dir/' + image_id

                # read the image
                image = cv2.imread(path)
           
                
                # convert to from BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # resize the image
                #image = resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH), mode='constant', preserve_range=True)
                
                # insert the image into X_train
                X_test[i] = image
                
             
            # Create Y_test
            # ===============
                
            for j, mask_id in enumerate(mask_id_list):

                # set the path to the mask
                path = 'mask_dir/' + mask_id

                # read the mask
                mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                
                # expand dims from (800,600) to (800,600,1)
                mask = np.expand_dims(mask, axis=-1)
                
                # resize the mask
                #mask = resize(mask, (IMAGE_HEIGHT, IMAGE_WIDTH), mode='constant', preserve_range=True)
                
                
                # insert the image into Y_train
                Y_test[j] = mask
            
            
            # Normalize the images
            X_test = X_test/255
            
            yield X_test, Y_test


# In[84]:


# Test the generator

# initialize
test_gen = test_generator(batch_size=5)

# run the generator
X_test, Y_test = next(test_gen)

print(X_test.shape)
print(Y_test.shape)


# In[85]:


# print the image from X_test

img = X_test[1,:,:,:]
plt.imshow(img)


# In[86]:


# print the mask from Y_test

msk = Y_test[1,:,:,0]
plt.imshow(msk)


# In[87]:


# Combine the mask and the image

plt.imshow(img, cmap='gray')
plt.imshow(msk, cmap='Blues', alpha=0.3)


# # Model Architecture

# In[99]:


import keras
from keras.layers import LSTM
from keras.models import Model, load_model
from keras.layers import Input, UpSampling2D
from keras.layers import Dropout, Lambda
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers. import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

from keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                        ModelCheckpoint, CSVLogger, LearningRateScheduler)


from keras.optimizers import Adam
from keras.losses import binary_crossentropy

from keras.initializers import he_normal 

import tensorflow as tf


# In[101]:


drop_out = 0.1
INIT_SEED = 101


# In[102]:


inputs = Input((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))

# Note: Tensorflow.js does not support lambda layers.
#s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer=he_normal(seed=INIT_SEED), padding='same') (inputs)
c1 = Dropout(drop_out) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer=he_normal(seed=INIT_SEED), padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer=he_normal(seed=INIT_SEED), padding='same') (p1)
c2 = Dropout(drop_out) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer=he_normal(seed=INIT_SEED), padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer=he_normal(seed=INIT_SEED), padding='same') (p2)
c3 = Dropout(drop_out) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer=he_normal(seed=INIT_SEED), padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer=he_normal(seed=INIT_SEED), padding='same') (p3)
c4 = Dropout(drop_out) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer=he_normal(seed=INIT_SEED), padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer=he_normal(seed=INIT_SEED), padding='same') (p4)
c5 = Dropout(drop_out) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer=he_normal(seed=INIT_SEED), padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer=he_normal(seed=INIT_SEED), padding='same') (u6)
c6 = Dropout(drop_out) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer=he_normal(seed=INIT_SEED), padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer=he_normal(seed=INIT_SEED), padding='same') (u7)
c7 = Dropout(drop_out) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer=he_normal(seed=INIT_SEED), padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer=he_normal(seed=INIT_SEED), padding='same') (u8)
c8 = Dropout(drop_out) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer=he_normal(seed=INIT_SEED), padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer=he_normal(seed=INIT_SEED), padding='same') (u9)
c9 = Dropout(drop_out) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer=he_normal(seed=INIT_SEED), padding='same') (c9)


outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy')

model.summary()


# # Create X_test

# In[103]:


# initialize
test_gen = test_generator(batch_size=len(df_test))

# run the generator
X_test, Y_test = next(test_gen)

print(X_test.shape)
print(Y_test.shape)


# # Train the Model

# In[104]:


num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = BATCH_SIZE
val_batch_size = BATCH_SIZE

# determine numtrain steps
train_steps = np.ceil(num_train_samples / train_batch_size)
# determine num val steps
val_steps = np.ceil(num_val_samples / val_batch_size)


# In[105]:


# Initialize the generators
train_gen = train_generator(batch_size=BATCH_SIZE)
val_gen = val_generator(batch_size=BATCH_SIZE)



filepath = "model.h5"

earlystopper = EarlyStopping(patience=5, verbose=1)

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, 
                                   verbose=1, mode='min')



log_fname = 'training_log.csv'
csv_logger = CSVLogger(filename=log_fname,
                       separator=',',
                       append=False)

callbacks_list = [checkpoint, earlystopper, csv_logger, reduce_lr]

history = model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=40, 
                              validation_data=val_gen, validation_steps=val_steps,
                             verbose=1,
                             callbacks=callbacks_list)


# In[118]:


import matplotlib.pyplot as plt

# Plotting Loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


plt.tight_layout()
plt.show()


# In[ ]:





# In[106]:


# Make a prediction

# initialize the test generator
test_gen = test_generator(batch_size=1)

model.load_weights('model.h5')
predictions = model.predict_generator(test_gen, 
                                      steps=len(df_test),  
                                      verbose=1)


# In[107]:


preds_test_thresh = (predictions >= 0.7).astype(np.uint8)

preds_test_thresh.shape

print(preds_test_thresh.min())
print(preds_test_thresh.max())


# In[108]:


# This is a predicted mask

mask = preds_test_thresh[3,:,:,0]
plt.imshow(mask, cmap='Reds', alpha=0.3)


# In[109]:


# This is a true mask

true_mask = Y_test[3,:,:,0]
plt.imshow(true_mask, cmap='Blues', alpha=0.3)


# In[110]:


# This is the x-ray image

image = X_test[3,:,:,:]

plt.imshow(image)


# In[111]:


# This is an overlay of the pred mask, true mask and 
# the x-ray image.

plt.imshow(image, cmap='gray')
plt.imshow(true_mask, cmap='Reds', alpha=0.3)
plt.imshow(mask, cmap='Blues', alpha=0.3)


# In[113]:


# set up the canvas for the subplots
plt.figure(figsize=(20,20))
plt.tight_layout()
plt.axis('Off')

predicted_masks = preds_test_thresh



    
# image
plt.subplot(1,4,1)
image = X_test[1,:,:,:] 
mask = predicted_masks[1, :, :, 0]
true_mask = Y_test[1, :, :, 0]
plt.imshow(image, cmap='gray')
plt.imshow(true_mask, cmap='Reds', alpha=0.3)
plt.imshow(mask, cmap='Blues', alpha=0.3)
plt.axis('off')


# image
plt.subplot(1,4,2)
image = X_test[2,:,:,:] 
mask = predicted_masks[2, :, :, 0]
true_mask = Y_test[2, :, :, 0]
plt.imshow(image, cmap='gray')
plt.imshow(true_mask, cmap='Reds', alpha=0.3)
plt.imshow(mask, cmap='Blues', alpha=0.3)
plt.axis('off')


# image
plt.subplot(1,4,3)
image = X_test[3,:,:,:]
mask = predicted_masks[3, :, :, 0]
true_mask = Y_test[3, :, :, 0]
plt.imshow(image, cmap='gray')
plt.imshow(true_mask, cmap='Reds', alpha=0.3)
plt.imshow(mask, cmap='Blues', alpha=0.3)
plt.axis('off')


# image
plt.subplot(1,4,4)
image = X_test[4,:,:,:] 
mask = predicted_masks[4, :, :, 0]
true_mask = Y_test[4, :, :, 0]
plt.imshow(image, cmap='gray')
plt.imshow(true_mask, cmap='Reds', alpha=0.3)
plt.imshow(mask, cmap='Blues', alpha=0.3)
plt.axis('off')



# ============ #


# set up the canvas for the subplots
plt.figure(figsize=(20,20))
plt.tight_layout()
plt.axis('Off')


# image
plt.subplot(1,4,1)
image = X_test[5,:,:,:] 
mask = predicted_masks[5, :, :, 0]
true_mask = Y_test[5, :, :, 0]
plt.imshow(image, cmap='gray')
plt.imshow(true_mask, cmap='Reds', alpha=0.3)
plt.imshow(mask, cmap='Blues', alpha=0.3)
plt.axis('off')


# image
plt.subplot(1,4,2)
image = X_test[6,:,:,:] 
mask = predicted_masks[6, :, :, 0]
true_mask = Y_test[6, :, :, 0]
plt.imshow(image, cmap='gray')
plt.imshow(true_mask, cmap='Reds', alpha=0.3)
plt.imshow(mask, cmap='Blues', alpha=0.3)
plt.axis('off')


# image
plt.subplot(1,4,3)
image = X_test[7,:,:,:] 
mask = predicted_masks[7, :, :, 0]
true_mask = Y_test[7, :, :, 0]
plt.imshow(image, cmap='gray')
plt.imshow(true_mask, cmap='Reds', alpha=0.3)
plt.imshow(mask, cmap='Blues', alpha=0.3)
plt.axis('off')


# image
plt.subplot(1,4,4)
image = X_test[8,:,:,:] 
mask = predicted_masks[8, :, :, 0]
true_mask = Y_test[8, :, :, 0]
plt.imshow(image, cmap='gray')
plt.imshow(true_mask, cmap='Reds', alpha=0.3)
plt.imshow(mask, cmap='Blues', alpha=0.3)
plt.axis('off')


# ============ #


# set up the canvas for the subplots
plt.figure(figsize=(20,20))
plt.tight_layout()
plt.axis('Off')


# image
plt.subplot(1,4,1)
image = X_test[9,:,:,:] 
mask = predicted_masks[9, :, :, 0]
true_mask = Y_test[9, :, :, 0]
plt.imshow(image, cmap='gray')
plt.imshow(true_mask, cmap='Reds', alpha=0.3)
plt.imshow(mask, cmap='Blues', alpha=0.3)
plt.axis('off')


# image
plt.subplot(1,4,2)
image = X_test[10,:,:,:] 
mask = predicted_masks[10, :, :, 0]
true_mask = Y_test[10, :, :, 0]
plt.imshow(image, cmap='gray')
plt.imshow(true_mask, cmap='Reds', alpha=0.3)
plt.imshow(mask, cmap='Blues', alpha=0.3)
plt.axis('off')


# image
plt.subplot(1,4,3)
image = X_test[11,:,:,:] 
mask = predicted_masks[11, :, :, 0]
true_mask = Y_test[11, :, :, 0]
plt.imshow(image, cmap='gray')
plt.imshow(true_mask, cmap='Reds', alpha=0.3)
plt.imshow(mask, cmap='Blues', alpha=0.3)
plt.axis('off')


# image
plt.subplot(1,4,4)
image = X_test[12,:,:,:] 
mask = predicted_masks[12, :, :, 0]
true_mask = Y_test[12, :, :, 0]
plt.imshow(image, cmap='gray')
plt.imshow(true_mask, cmap='Reds', alpha=0.3)
plt.imshow(mask, cmap='Blues', alpha=0.3)
plt.axis('off')


plt.show()


# In[131]:


from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'true_labels' and 'predicted_labels' are binary arrays (0 or 1)
true_labels = Y_test.reshape(-1)  # Flatten the true masks
predicted_labels = (predicted_masks > threshold).reshape(-1)  # Flatten the predicted masks

# Create a confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Calculate overall accuracy
accuracy = accuracy_score(true_labels, predicted_labels) * 100

# Extract values from the confusion matrix
tn, fp, fn, tp = cm.ravel()

# Calculate percentages
total_samples = len(true_labels)
false_positive_rate = fp / (fp + tn) * 100
false_negative_rate = fn / (fn + tp) * 100

# Plot the percentages
labels = ['Correct Predictions', 'False Positives', 'False Negatives']
percentages = [accuracy, false_positive_rate, false_negative_rate]

plt.figure(figsize=(8, 6))
sns.barplot(x=labels, y=percentages, palette='Blues')
plt.title(f'Model Performance: Accuracy = {accuracy:.2f}%')
plt.ylabel('Percentage')
plt.show()


# In[ ]:




