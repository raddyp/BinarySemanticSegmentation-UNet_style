# -*- coding: utf-8 -*-
"""
Binary Semantic Segmentation example based on U-Net on Forest cover dataset
"""

# %%

import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import gc
import cv2
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.models import save_model,load_model

# display plots of multiple images side by side
def display(display_list):
    plt.figure(figsize=(15, 15))    
    title = ['Input image', 'True mask', 'Predicted mask']
    
    for i in range(len(display_list)):
      plt.subplot(1, len(display_list), i+1)
      plt.title(title[i])
      plt.imshow((display_list[i]))
      plt.axis('off')
    plt.show()
  
 
# read image data from folders, resize and vectorize them
def process_data_from_folder(src_dir):    
        
    data_dir = os.path.join(src_dir,'images')
    masks_dir = os.path.join(src_dir,'masks')    
    
    # Images in dataset are 256x256x3. too big for the machine to process. 
    # Reducing the image size to 128x128x3
 
    width = 128
    height = 128  
    
    data = []
    masks = []
    
    # read images from folder
    path = os.path.join(data_dir)    
    files = os.listdir(path)     
    
    for file in files:
        # print(i,'   ',file)
        filename = os.path.join(path, file)
        if (os.path.isdir(filename)):
            #print(filename, "  is directory")
            continue
        if(os.path.getsize(filename) ==0):
            print(file, "  is zero length, so ignoring")
            continue
        
        # read and resize image
        image = cv2.imread(filename)    
        res_im = cv2.resize(image,(height,width))   
        data.append((res_im))
       
    # read masks from folder
    path = os.path.join(masks_dir)    
    files = os.listdir(path)       
    
    for file in files:
        # print(file)
        filename = os.path.join(path, file)
        if (os.path.isdir(filename)):
            #print(filename, "  is directory")
            continue
        if(os.path.getsize(filename) ==0):
            print(file, "  is zero length, so ignoring")
            continue
        
        # read and resize masks
        image = cv2.imread(filename,0)   
        res_im = cv2.resize(image,(height,width))     
        masks.append((res_im))
    
    data = np.array(data)
    masks = np.array(masks)   
        
    return data, masks
 
# read image datas from h5 file
def read_data(filename):
        
    hfile = h5py.File(filename,'r')    
    data = np.array(hfile.get('data'))
    masks = np.array(hfile.get('masks'))
    hfile.close()
    
    return data,masks


# split data into train-test split
def split_data(data, masks):
    
    # randomize distribution to avoid biases
    randomize = np.arange(len(data))
    np.random.shuffle(randomize)
    data = data[randomize]
    masks = masks[randomize]
    
    split_size = .8,.1,.1
    r1 = int(len(data)*split_size[0])
    r2 = int(len(data)*split_size[1])
    r3 = int(len(data)*split_size[2])
    
    x_train,x_val,x_test = data[0:r1,:],data[r1:r1+r2,:],data[r1+r2:r1+r2+r3,:]
    y_train,y_val,y_test = masks[0:r1,:],masks[r1:r1+r2,:],masks[r1+r2:r1+r2+r3,:]         
    
    # normalize data
    x_train = x_train.astype('float32')/255 
    x_val = x_val.astype('float32')/255
    x_test = x_test.astype('float32')/255

    y_train = y_train.astype('float32')/255
    y_val = y_val.astype('float32')/255
    y_test = y_test.astype('float32')/255

    return x_train,y_train,x_val,y_val,x_test,y_test

# create the U-Net model
def create_model(img_size,num_classes):
    
    # example implementation of U-Net architecture.
    # parameters have been changed slightly from the original paper to suit this dataset
    
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0,seed=None)
    num_filters = 16
    inputs= tfl.Input(shape=img_size)
      
    # downsampling section
    x = tfl.Conv2D(filters = num_filters,kernel_size = (3,3),activation='relu',padding = 'same',kernel_initializer = initializer)(inputs)      
    x = tfl.Conv2D(filters = num_filters,kernel_size = (3,3),activation='relu',padding = 'same',kernel_initializer = initializer)(x)  
    x = tfl.Dropout(0.3)(x)
    x = tfl.BatchNormalization()(x)

    skip1 = x 
    x = tfl.MaxPooling2D(pool_size=(2, 2))(x) 

    x = tfl.Conv2D(filters = 2*num_filters,kernel_size = (3,3),activation='relu',padding = 'same',kernel_initializer = initializer)(x)    
    x = tfl.Conv2D(filters = 2*num_filters,kernel_size = (3,3),activation='relu',padding = 'same',kernel_initializer = initializer)(x)  
    x = tfl.Dropout(0.3)(x)
    x = tfl.BatchNormalization()(x)

    skip2 =x 
    x = tfl.MaxPooling2D(pool_size=(2, 2))(x)

    x = tfl.Conv2D(filters = 4*num_filters,kernel_size = (3,3),activation='relu',padding = 'same',kernel_initializer = initializer)(x)    
    x = tfl.Conv2D(filters = 4*num_filters,kernel_size = (3,3),activation='relu',padding = 'same',kernel_initializer = initializer)(x)  
    x = tfl.Dropout(0.3)(x)
    x = tfl.BatchNormalization()(x)

    skip3= x 
    x = tfl.MaxPooling2D(pool_size=(2, 2))(x)

    x = tfl.Conv2D(filters = 8*num_filters,kernel_size = (3,3),activation='relu',padding = 'same',kernel_initializer = initializer)(x)        
    x = tfl.Conv2D(filters = 8*num_filters,kernel_size = (3,3),activation='relu',padding = 'same',kernel_initializer = initializer)(x)  
    x = tfl.Dropout(0.3)(x)
    x = tfl.BatchNormalization()(x)

    skip4 = x
    x = tfl.MaxPooling2D(pool_size=(2, 2))(x)

    x = tfl.Conv2D(filters = 16*num_filters,kernel_size = (3,3),activation='relu',padding = 'same',kernel_initializer = initializer)(x)      
    x = tfl.Conv2D(filters = 16*num_filters,kernel_size = (3,3),activation='relu',padding = 'same',kernel_initializer = initializer)(x)  
    x = tfl.Dropout(0.3)(x)
    x = tfl.BatchNormalization()(x)
    
    
    # upsampling section
    x = tfl.Conv2DTranspose(filters = 8*num_filters,kernel_size = (3,3),strides = (2,2),padding = 'same',kernel_initializer = initializer)(x)
    x = tfl.Concatenate()([x,skip4])  
    x = tfl.Conv2D(filters = 8*num_filters,kernel_size = (3,3),strides = (1,1),padding = 'same',kernel_initializer = initializer)(x)
    x = tfl.Conv2D(filters = 8*num_filters,kernel_size = (3,3),strides = (1,1),padding = 'same',kernel_initializer = initializer)(x)
    x = tfl.Dropout(0.3)(x)
    x = tfl.BatchNormalization()(x)

    x = tfl.Conv2DTranspose(filters = 4*num_filters,kernel_size = (3,3),strides = (2,2),padding = 'same',kernel_initializer = initializer)(x)
    x = tfl.Concatenate()([x,skip3]) 
    x = tfl.Conv2D(filters = 4*num_filters,kernel_size = (3,3),strides = (1,1),padding = 'same',kernel_initializer = initializer)(x)
    x = tfl.Conv2D(filters = 4*num_filters,kernel_size = (3,3),strides = (1,1),padding = 'same',kernel_initializer = initializer)(x)
    x = tfl.Dropout(0.3)(x)
    x = tfl.BatchNormalization()(x)

    x = tfl.Conv2DTranspose(filters = 2*num_filters,kernel_size = (3,3),strides = (2,2),padding = 'same',kernel_initializer = initializer)(x)
    x = tfl.Concatenate()([x,skip2])
    x = tfl.Conv2D(filters = 2*num_filters,kernel_size = (3,3),strides = (1,1),padding = 'same',kernel_initializer = initializer)(x)
    x = tfl.Conv2D(filters = 2*num_filters,kernel_size = (3,3),strides = (1,1),padding = 'same',kernel_initializer = initializer)(x)
    x = tfl.Dropout(0.3)(x)
    x = tfl.BatchNormalization()(x)

    x = tfl.Conv2DTranspose(filters = num_filters,kernel_size = (3,3),strides = (2,2),padding = 'same',kernel_initializer = initializer)(x)
    x = tfl.Concatenate()([x,skip1]) 
    x = tfl.Conv2D(filters = num_filters,kernel_size = (3,3),strides = (1,1),padding = 'same',kernel_initializer = initializer)(x)
    x = tfl.Conv2D(filters = num_filters,kernel_size = (3,3),strides = (1,1),padding = 'same',kernel_initializer = initializer)(x)
    x = tfl.Dropout(0.3)(x)
    x = tfl.BatchNormalization()(x)

    outputs = tfl.Conv2D(num_classes,1,activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                    )

    model.summary()
    
    return model

def compute_metrics(y_test,predictions):
        
    # NN output is mxhxwx1, reshaping it to mxhxw
    m,r,c,ch = np.shape(predictions)
    predictions = np.reshape(predictions, (m,r,c))
    
    # checking for normalization errors and binarizing the results 
    maxp,maxg = np.max(predictions),np.max(y_test)
    
    if (maxg == 1.0):
        y_test = (y_test>0.5).astype(np.int16)
    elif (maxg==255.0):
        y_test = y_test/255.0
        y_test = (y_test>0.5).astype(np.int16)
    
    if (maxp == 1.0):
        predictions = (predictions>0.5).astype(np.int16)
    elif (maxp==255.0):
        predictions = predictions/255.0
        predictions = (predictions>0.5).astype(np.int16)
    
    # computing IOU scores
    error = []
    for i in range (len(predictions)):
        
        # predicted segmentation
        s = predictions[i]
        # groundtruth
        gt = y_test[i]    
           
        intersection = np.sum( np.logical_and(s,gt))
        union = np.sum( np.logical_or(s,gt))
        iou = intersection/union
        error.append(iou)
    
    error = np.array(error)
    
    # find the predictions with the high scores    
    good_res = np.sum((error>0.9))
    print('number of good results  ',good_res,'/',len(predictions))
    
    return

def visualize_plots(history):
    # visualize plots

    acc      = history.history[     'accuracy' ]
    val_acc  = history.history[ 'val_accuracy' ]
    loss     = history.history[    'loss' ]
    val_loss = history.history['val_loss' ]
    epochs   = range(len(acc)) # Get number of epochs

    #------------------------------------------------
    # Plot training and validation accuracy per epoch
    #------------------------------------------------
    plt.plot  ( epochs,     acc, label = 'Train Acc' )
    plt.plot  ( epochs, val_acc, label = 'Val Acc' )
    plt.title ('Training and validation accuracy')
    plt.xlabel ('epochs')
    plt.legend(loc="upper right")
    plt.figure()

    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    plt.plot  ( epochs,     loss,label = 'Train loss' )
    plt.plot  ( epochs, val_loss , label = 'Val loss')
    plt.title ('Training and validation loss'   )
    plt.xlabel ('epochs')
    plt.legend(loc="lower right")

    return
# %%

gc.enable()

#process images from directory 
# src_dir = 'D:/Data/forest_aerial_seg/'     
# data,masks = process_data_from_folder(src_dir) 

# # save data to h5 file format
# trf = h5py.File('D:/Data/forest_aerial_seg/forest_seg.h5','w')
# trf.create_dataset('data',data=data)
# trf.create_dataset('masks',data=masks)
# trf.close()

# read form h5 to verify if all data is written
data,masks = read_data('D:/Data/forest_aerial_seg/forest_seg.h5')

#split data into train, validation & test sets (0.8:0.1:0.1)
x_train,y_train,x_val,y_val,x_test,y_test=split_data(data, masks)

# free up memory
del data,masks

# get image dimensions
m,h,w,c = np.shape(x_train)

# create the U-Net segmentation model
num_classes = 1
model = create_model((h,w,c),num_classes)

# Train your model
# this is a relatively big model with large data size, to make it work on machine, batch size is reduced
history = model.fit(x_train,y_train, 
                    epochs=100, 
                    batch_size=16,
                    verbose = 1, 
                    validation_data=(x_val,y_val),
                    )

# save model to directory
# save_model(model,'D:/Analysis/forest_seg/seg.h5')

# load model form directory
# model = load_model('D:/Analysis/forest_seg/seg.h5')

# predictions on test set
pred = model.predict(x_test)
score = model.evaluate(x_test, y_test, verbose = 0) 

print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

# compute metrics
compute_metrics(y_test,pred)

#  visualize training acuracy & loss plots
visualize_plots(history)

# plot y_test prediction results in a grid
pred_copy = (pred>0.5).astype(np.uint8)
pred_copy = pred_copy*255

i=35
x = x_test[i,:,:,:]
y = y_test[i,:,:]
p = pred_copy[i]
display([x,y,p])


#%%
# predictions on single images demo

img = cv2.imread('D:/Data/test1.jpg',1)
resize_img = cv2.resize(img,(128,128))     
norm_img = np.reshape(np.array(resize_img/255.),(1,128,128,3))

prediction = model.predict(norm_img)
prediction = (prediction>0.5).astype(np.uint8)
prediction = prediction*255
prediction = np.reshape(prediction,(128,128))

plt.subplot(1, 2, 1)
plt.title('Test image')
plt.imshow((img))
plt.subplot(1, 2, 2)
plt.title('Prediction')
plt.imshow(prediction)

plt.axis('off')
plt.show()

gc.collect()
