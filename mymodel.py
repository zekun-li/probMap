import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras                              
from keras.models import Sequential  
from keras.layers import Dense, Dropout, Flatten 
from keras.layers import Conv2D, MaxPooling2D    
from keras import backend as K                   
from keras.callbacks import Callback             
from keras.layers import Lambda, Input, Dense, Concatenate ,Conv2DTranspose 
from keras.layers import LeakyReLU,BatchNormalization,AveragePooling2D,Reshape 
from keras.layers import UpSampling2D,ZeroPadding2D
from keras.losses import mse, binary_crossentropy                           
from keras.models import Model                                                     
from keras.preprocessing.image import apply_transform, transform_matrix_offset_center
from sklearn.model_selection import train_test_split                                                                                                              
import numpy as np  
import cv2
import argparse     
import glob
import tensorflow as tf 


# In[2]:
if_rot_aug = False
myvalidation_steps = 10

# original stable version
def myModel1000_0():
    input_shape = (1000,1000,3)
    inputs = Input(shape=input_shape, name='input')    
    x = MaxPooling2D((4,4))(inputs)

    x = Conv2D(8, kernel_size = (10,10), activation = 'relu', padding='same')(x)
    #x = Conv2D(8, kernel_size = (10,10), activation = 'relu', padding='same')(x)

    x = Conv2D(64, kernel_size = (5,5), activation = 'relu', padding='same')(x)
    #x = Conv2D(64, kernel_size = (3,3), activation = 'relu')(x)

    x = MaxPooling2D((4,4))(x)


    x = Conv2D(128, kernel_size = (5,5), activation = 'relu', padding='same')(x)
    #x = Conv2D(32, kernel_size = (3,3), activation = 'relu')(x)

    x = MaxPooling2D((4,4))(x)

    x = Conv2D(256, kernel_size = (5,5), activation = 'relu', padding='same')(x)
    #x = Conv2D(64, kernel_size = (3,3), activation = 'relu')(x)

    x = UpSampling2D((4,4))(x)

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(64, kernel_size = (5,5), activation = 'relu', padding='same')(x)

    x = UpSampling2D((4,4))(x)

    x = Conv2D(2, kernel_size = (10,10), activation = 'softmax', padding='same')(x)

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = UpSampling2D((4,4))(x)

    output= x

    model = Model(inputs, output, name = 'zekun_model')
    model.summary()
    return model

def myModel1000_1():
    input_shape = (1000,1000,3)
    inputs = Input(shape=input_shape, name='input')    
    x = MaxPooling2D((2,2))(inputs)

    x = Conv2D(8, kernel_size = (10,10), activation = 'relu', padding='same')(x)
    #x = Conv2D(8, kernel_size = (10,10), activation = 'relu', padding='same')(x)

    x = Conv2D(64, kernel_size = (5,5), activation = 'relu', padding='same')(x)
    #x = Conv2D(64, kernel_size = (3,3), activation = 'relu')(x)

    x = MaxPooling2D((4,4))(x)


    x = Conv2D(128, kernel_size = (5,5), activation = 'relu', padding='same')(x)
    #x = Conv2D(32, kernel_size = (3,3), activation = 'relu')(x)

    x = MaxPooling2D((4,4))(x)

    x = Conv2D(256, kernel_size = (5,5), activation = 'relu', padding='same')(x)
    #x = Conv2D(64, kernel_size = (3,3), activation = 'relu')(x)

    x = UpSampling2D((4,4))(x)

    
    x = Conv2D(64, kernel_size = (5,5), activation = 'relu', padding='same')(x)
    
    
    x = UpSampling2D((4,4))(x)
    x = ZeroPadding2D(padding=(2, 2))(x)

    x = Conv2D(2, kernel_size = (10,10), activation = 'softmax', padding='same')(x)

    #x = ZeroPadding2D(padding=(2, 2))(x)
    x = UpSampling2D((2,2))(x)

    output= x

    model = Model(inputs, output, name = 'zekun_model')
    model.summary()
    return model



# In[27]:

# kernel size (3,3)
def myModel1000():
    input_shape = (1000,1000,3)
    inputs = Input(shape=input_shape, name='input')    
    x = AveragePooling2D((2,2))(inputs)

    x = Conv2D(8, kernel_size = (10,10), activation = 'relu', padding='same')(x)
    #x = Conv2D(8, kernel_size = (10,10), activation = 'relu', padding='same')(x)

    x = Conv2D(64, kernel_size = (3,3), activation = 'relu', padding='same')(x)
    #x = Conv2D(64, kernel_size = (3,3), activation = 'relu')(x)

    x =AveragePooling2D((4,4))(x)


    x = Conv2D(128, kernel_size = (3,3), activation = 'relu', padding='same')(x)
    #x = Conv2D(32, kernel_size = (3,3), activation = 'relu')(x)

    x = AveragePooling2D((4,4))(x)

    x = Conv2D(256, kernel_size = (3,3), activation = 'relu', padding='same')(x)
    #x = Conv2D(64, kernel_size = (3,3), activation = 'relu')(x)

    x = UpSampling2D((4,4))(x)

    
    x = Conv2D(64, kernel_size = (3,3), activation = 'relu', padding='same')(x)
    
    
    x = UpSampling2D((4,4))(x)
    x = ZeroPadding2D(padding=(2, 2))(x)

    x = Conv2D(2, kernel_size = (10,10), activation = 'softmax', padding='same')(x)

    #x = ZeroPadding2D(padding=(2, 2))(x)
    x = UpSampling2D((2,2))(x)

    output= x

    model = Model(inputs, output, name = 'zekun_model')
    model.summary()
    return model



class DataGenerator(object):   
    # image_root_path : map and mask images dir
    # list_path points to the file-list of train/val/test split 

    def __init__(self, image_root_path, list_path, batch_size = 128,  seed = 1234, mode = 'training' ):  
        # get the filenames of map files
        f = open(list_path,'r') 
        file_list = f.read().splitlines() # get the list of file names [['101201496.jpg','102903919.jpg',...]
        f.close()
        
        # get the full path of map & maks files
        X,Y = [],[]
        for file_name in file_list:
            # map images
            x = glob.glob( image_root_path + '/' + file_name.split('.jpg')[0] + '*.jpg'  )
            X.extend(x)
            
            # mask images (should search for corresponding images, NOT to use glob AGAIN!)
            # mask images
            y = []
            for patch_path in x:
                # patch_path eg: '/nas/home/zekunl/dornsife/sub_maps_masks/101201496_h9w8.jpg'
                base_name = os.path.basename(patch_path) # eg: 101201496_h9w8.jpg 
                this_mask_name = image_root_path + '/' + 'masked_' + base_name 
                y.append(this_mask_name)
            Y.extend(y) 

        print 'num_samples = ',len(X)
        # X: map file list, Y: binary mask file list                                                                                                             
        self.idx = 0 
        self.nb_samples =  len(X)   
        self.X = X                 
        self.Y = Y                 
        self.batch_size = batch_size 
                                                                                                                                                                        
        self.seed = seed             
        self.mode = mode                                    
        np.random.seed(seed)         
 
    def __getitem__(self, batch_idx):   
        # randomly shuffle for training split
        # sequentiallyy take the testing data
        #print self.mode, batch_idx
        if (self.mode == 'training'):     
            sample_indices = np.random.randint( 0, self.nb_samples, self.batch_size )            
        else:                                                                                    
            # batch_idx keeps increasing regardless of epoch (only depend on number of updates), 
            # we need to reset batch_idx for each epoch to make sure validaton data is the same across different epochs
            # we are doing the resetting using modulo operation
            batch_idx = batch_idx % myvalidation_steps
            sample_indices = range( batch_idx * self.batch_size, min( self.nb_samples, (batch_idx+1) * self.batch_size ) )    

        # get the file paths
        subset_X = [self.X[i] for i in sample_indices]
        subset_Y = [self.Y[i] for i in sample_indices]
                                                                                                                                                                  
        
        # get the images
        batch_X = []    
        batch_Y = []
        for map_path,mask_path in zip(subset_X,subset_Y):        
            # rad map and masks
            map_img =  cv2.imread(map_path) 
            map_img = map_img / 255.
            
            mask_img = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE) # read grayscale, 
            
            thresh = 100 # (values are only 0 and 150)
            bw_img = cv2.threshold(mask_img, thresh, 255, cv2.THRESH_BINARY)[1]
            proba_map = bw_img / 255.   # 0s and 1s
            proba_map = np.expand_dims(proba_map, axis = -1) # convert 2d to 3d

            if if_rot_aug == True:
                if self.mode == 'training':
                    # piece of code taken from 
                    # https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image.py
                    theta = np.pi / 180 * np.random.uniform(-90, 90)
                    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], 
                                                [np.sin(theta), np.cos(theta), 0],
                                                [0, 0, 1]])
                    h, w = map_img.shape[0], map_img.shape[1] # h,w,c order 
                    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)

                    # Do the actual rotation!
                    map_img = apply_transform(map_img, transform_matrix, channel_axis = 2, fill_mode='nearest', cval=0.) # cval is dummy  
                    proba_map = apply_transform(proba_map, transform_matrix, channel_axis = 2, fill_mode='constant', cval=0.) # avoid erroneous `false positive`

            proba_map_reverse = 1 - proba_map
            proba_map = np.concatenate([proba_map, proba_map_reverse],axis = -1)
            
            batch_X.append(map_img)
            batch_Y.append(proba_map)
            
        batch_X = np.array(batch_X) + 1e-5 # avoid 0 values
        batch_Y = np.array(batch_Y) 
        
        
        return batch_X, batch_Y
                                                                                                  
                                                                                                  
    def next(self):                                                                               
        idx = self.idx                                                                            
        self.idx = ( 1 + idx ) % self.nb_samples                                                  
        return  self[idx] 


# In[28]:

#image_root_path = '/nas/home/zekunl/dornsife/sub_maps_masks'

image_root_path = os.environ['TMPDIR'] + '/sub_maps_masks'

model = myModel1000()
#model.compile('adam',loss = 'binary_crossentropy')
model.compile('adam',loss = 'categorical_crossentropy')

train_datagen = DataGenerator(image_root_path = image_root_path ,list_path = '/nas/home/zekunl/dornsife/train_val_split/USGS_train.list', batch_size= 12,  seed = 1234, mode = 'training')
val_datagen = DataGenerator(image_root_path = image_root_path ,list_path = '/nas/home/zekunl/dornsife/train_val_split/USGS_val.list', batch_size= 12,  seed = 1234, mode = 'validation')

ith = '04_avgpool'
out_log_name = 'attempt_' + ith + '.csv'
csv_logger = keras.callbacks.CSVLogger(out_log_name) 
check_point = keras.callbacks.ModelCheckpoint('attempt_' + ith + '{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=False, save_weights_only=False, period = 2)
callbacks = [csv_logger, check_point]


'''
x,y = train_datagen[0]
x1,y1 = val_datagen[1]
mydict = {'x':x,'y':y,'x1':x1, 'y1':y1}
import pickle
with open('data.pkl','w') as f:
    pickle.dump(mydict, f)
'''


model.fit_generator(train_datagen, steps_per_epoch = 100, epochs = 10, validation_data = val_datagen, validation_steps = myvalidation_steps, callbacks = callbacks) 
model.save('attempt_' + ith + '.hdf5')
