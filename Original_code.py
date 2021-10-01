#!/usr/bin/env python
# coding: utf-8

# In[83]:


import os, shutil
from keras import optimizers
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D


# In[84]:


original_datasets_dir_normal = r'F:/Kaggle/Pneumonia-chest-xray/data/NORMAL/'
original_datasets_dir_pneumonia = r'F:/Kaggle/Pneumonia-chest-xray/data/PNEUMONIA/'

base_dir = r'F:/Kaggle/Pneumonia-chest-xray/data/'
if not os.path.isdir(base_dir): os.mkdir(base_dir)


# In[85]:


train_dir = os.path.join(base_dir,'train')
if not os.path.isdir(train_dir): os.mkdir(train_dir)
val_dir = os.path.join(base_dir,'val')
if not os.path.isdir(val_dir): os.mkdir(val_dir)
test_dir = os.path.join(base_dir,'test')
if not os.path.isdir(test_dir): os.mkdir(test_dir)


# In[86]:


train_normal_dir = os.path.join(train_dir,'NORMAL')
if not os.path.isdir(train_normal_dir):
    os.mkdir(train_normal_dir)
train_pneumonia_dir = os.path.join(train_dir,'PNEUMONIA')
if not os.path.isdir(train_pneumonia_dir):
    os.mkdir(train_pneumonia_dir)
test_normal_dir = os.path.join(test_dir,'NORMAL')
if not os.path.isdir(test_normal_dir):
    os.mkdir(test_normal_dir)
test_pneumonia_dir = os.path.join(test_dir,'PNEUMONIA')
if not os.path.isdir(test_pneumonia_dir):
    os.mkdir(test_pneumonia_dir)
val_normal_dir = os.path.join(val_dir,'NORMAL')
if not os.path.isdir(val_normal_dir):
    os.mkdir(val_normal_dir)
val_pneumonia_dir = os.path.join(val_dir,'PNEUMONIA')
if not os.path.isdir(val_pneumonia_dir):
    os.mkdir(val_pneumonia_dir)


# In[87]:


fnames = ['1 ({}).jpeg'.format(i) for i in range(1,1001)]
for fname in fnames:
    src = os.path.join(original_datasets_dir_normal, fname)
    dst = os.path.join(train_normal_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['1 ({}).jpeg'.format(i) for i in range(1001,1301)]
for fname in fnames:
    src = os.path.join(original_datasets_dir_normal, fname)
    dst = os.path.join(val_normal_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['1 ({}).jpeg'.format(i) for i in range(1301,1501)]
for fname in fnames:
    src = os.path.join(original_datasets_dir_normal, fname)
    dst = os.path.join(test_normal_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['2 ({}).jpeg'.format(i) for i in range(1,1001)]
for fname in fnames:
    src = os.path.join(original_datasets_dir_pneumonia, fname)
    dst = os.path.join(train_pneumonia_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['2 ({}).jpeg'.format(i) for i in range(1001,1301)]
for fname in fnames:
    src = os.path.join(original_datasets_dir_pneumonia, fname)
    dst = os.path.join(val_pneumonia_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['2 ({}).jpeg'.format(i) for i in range(1301,1501)]
for fname in fnames:
    src = os.path.join(original_datasets_dir_pneumonia, fname)
    dst = os.path.join(test_pneumonia_dir, fname)
    shutil.copyfile(src, dst)


# In[88]:


train_datasets_normal = r'F:/F:/Kaggle/Pneumonia-chest-xray/data/train/NORMAL/'
train_datasets_pneumonia = r'/F:/Kaggle/Pneumonia-chest-xray/data/train/PNEUMONIA/'
val_datasets_normal = r'F:/F:/Kaggle/Pneumonia-chest-xray/data/val/NORMAL/'
val_datasets_pneumonia = r'/F:/Kaggle/Pneumonia-chest-xray/data/val/PNEUMONIA/'
test_datasets_normal = r'F:/F:/Kaggle/Pneumonia-chest-xray/data/test/NORMAL/'
test_datasets_pneumonia = r'/F:/Kaggle/Pneumonia-chest-xray/data/test/PNEUMONIA/'

train_normal = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
train_pneumonia = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
all_train_contents = train_normal + train_pneumonia
print("all_train_contents: ", all_train_contents)

val_normal = len(os.listdir(os.path.join(val_dir, 'NORMAL')))
val_pneumonia = len(os.listdir(os.path.join(val_dir, 'PNEUMONIA')))
all_val_contents = val_normal + val_pneumonia
print("all_val_contents: ", all_val_contents)

test_normal = len(os.listdir(os.path.join(test_dir, 'NORMAL')))
test_pneumonia = len(os.listdir(os.path.join(test_dir, 'PNEUMONIA')))
all_test_contents = test_normal + test_pneumonia
print("all_test_contents: ", all_test_contents)

all_normal = train_normal + val_normal + test_normal
print("all_normal: ", all_normal)
all_pneumonia = train_pneumonia + val_pneumonia + test_pneumonia
print("all_pneumonia: ", all_pneumonia)


# In[89]:


print('Train Normal images: ', len(os.listdir(train_normal_dir)))
print('Train Pneumonia images: ', len(os.listdir(train_pneumonia_dir)))

print('Val Normal images: ', len(os.listdir(val_pneumonia_dir)))
print('Val Pneumonia images: ', len(os.listdir(val_pneumonia_dir)))

print('Test Normal images: ', len(os.listdir(test_normal_dir)))
print('Test Pneumonia images: ', len(os.listdir(test_pneumonia_dir)))


# In[101]:


model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))

model.summary()


# In[102]:


model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy',
                      tf.keras.metrics.Precision(name='precision'),
                      tf.keras.metrics.Recall(name='recall')])


# In[103]:


train_generator = ImageDataGenerator(rescale=1./255)
val_generator = ImageDataGenerator(rescale=1./255)
test_generator = ImageDataGenerator(rescale=1./255)

train_generator = train_generator.flow_from_directory(train_dir, 
                                                      target_size=(150,150), 
                                                      batch_size=20, 
                                                      class_mode='categorical')
val_generator = val_generator.flow_from_directory(val_dir, 
                                                   target_size=(150,150), 
                                                   batch_size=6, 
                                                   class_mode='categorical')
test_generator = test_generator.flow_from_directory(test_dir, 
                                                   target_size=(150,150), 
                                                   batch_size=4, 
                                                   class_mode='categorical')


# In[104]:


for data_batch, labels_batch in train_generator:
    print('data batch shape: ',data_batch.shape)
    print('labels batch shape: ',labels_batch.shape)
    break


# In[105]:


history = model.fit(train_generator, 
                    steps_per_epoch=100, 
                    epochs=100, 
                    validation_data=val_generator, 
                    validation_steps=50)


# In[106]:


fig, ax = plt.subplots(1, 4, figsize=(20, 3))
ax = ax.ravel()

for i, met in enumerate(['precision', 'recall', 'accuracy', 'loss']):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])


# In[107]:


loss, acc, prec, rec = model.evaluate(test_generator)


# In[110]:


model.save('Pneumonia-chest-xray.h5')

