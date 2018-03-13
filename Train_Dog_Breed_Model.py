
# coding: utf-8

# In[23]:


from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers import Input, Dense, Flatten
from keras.layers.core import Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing import image  
from keras.utils import np_utils
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_files
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm   
import numpy as np
from glob import glob
import os.path
import pandas as pd
import pickle


# In[2]:


# Download file if not already in directory
# Go to https://www.kaggle.com/c/dog-breed-identification/data download and unpack train.zip and labels.csv.zip


# In[3]:


try:
    assert os.path.isdir('dogImages/train') and os.path.isdir('dogImages/test') and os.path.isdir('dogImages/valid')
except:
    print("Download the images from https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip and unpack.")
    raise


# In[4]:


def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


# In[5]:


train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')


# In[6]:


dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]


# In[7]:


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(299, 299))
    # convert PIL.Image.Image type to 3D tensor with shape (299, 299, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 299, 299, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


# In[8]:


from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')
valid_tensors = paths_to_tensor(valid_files).astype('float32')
test_tensors = paths_to_tensor(test_files).astype('float32')


# In[9]:


inceptionV3 = InceptionV3(weights='imagenet', include_top=False)
# Inception Model
train_preprocessed_input = preprocess_input(train_tensors)
train_preprocessed_tensors = inceptionV3.predict(train_preprocessed_input, batch_size=32)
print("InceptionV3 TrainSet shape", train_preprocessed_tensors.shape[1:])
test_preprocessed_input = preprocess_input(test_tensors)
test_preprocessed_tensors = inceptionV3.predict(test_preprocessed_input, batch_size=32)
print("InceptionV3 TestSet shape", test_preprocessed_tensors.shape[1:])
valid_preprocessed_input = preprocess_input(valid_tensors)
valid_preprocessed_tensors = inceptionV3.predict(valid_preprocessed_input, batch_size=32)
print("InceptionV3 ValidSet shape", valid_preprocessed_tensors.shape[1:])


# In[21]:


net_input = Input(shape=(8, 8, 2048))
net = GlobalAveragePooling2D()(net_input)
net = Dense(512, use_bias=False, kernel_initializer='uniform')(net)
net = BatchNormalization()(net)
net = Activation("relu")(net)
net = Dropout(0.5)(net)
net = Dense(256, use_bias=False, kernel_initializer='uniform')(net)
net = BatchNormalization()(net)
net = Activation("relu")(net)
net = Dropout(0.5)(net)
net = Dense(133, kernel_initializer='uniform', activation="softmax")(net)

dog_breed_model = Model(inputs=[net_input], outputs=[net])
dog_breed_model.summary()


# In[22]:


dog_breed_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-04), metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='dogbreed_model.hdf5', verbose=1, save_best_only=True)
dog_breed_model.fit([train_preprocessed_tensors], train_targets, 
          validation_data=([valid_preprocessed_tensors], valid_targets),
          epochs=50, batch_size=4, callbacks=[checkpointer], verbose=1)


# In[26]:


dog_breed_model.load_weights('dogbreed_model.hdf5') # in case you haven't train it 
predictions = dog_breed_model.predict([test_preprocessed_tensors])
breed_predictions = [np.argmax(prediction) for prediction in predictions]
breed_true_labels = [np.argmax(true_label) for true_label in test_targets]
print('Test accuracy: %.4f%%' % (accuracy_score(breed_true_labels, breed_predictions) * 100))


# In[28]:


pickle.dump(dog_names, open('dogbreed_labels.pickle', 'wb'))

