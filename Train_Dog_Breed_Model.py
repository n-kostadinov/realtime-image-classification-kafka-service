
# coding: utf-8

# In[1]:


from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers import Input, Dense
from keras.layers.core import Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing import image                  
from tqdm import tqdm   
from keras.utils import np_utils
import numpy as np
from glob import glob
import os.path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import pickle


# In[2]:


# Download file if not already in directory
# Go to https://www.kaggle.com/c/dog-breed-identification/data download and unpack train.zip and labels.csv.zip


# In[3]:


try:
    assert os.path.isfile('labels.csv') and os.path.isdir('train')
except:
    print("Go to https://www.kaggle.com/c/dog-breed-identification/data download and unpack train.zip and labels.csv.zip")
    raise


# In[4]:


file_paths = glob("train/*")
print('Dog files:', len(file_paths))

labels = pd.read_csv('labels.csv')
dog_labels_mapping = dict(zip(labels.id.values, labels.breed.values))
dog_label_encoder = LabelEncoder().fit(labels.breed.unique())
dog_labels = [dog_labels_mapping[path.replace('train/', '').replace('.jpg','')] for path in file_paths]
encoded_dog_labels = dog_label_encoder.transform(dog_labels)
dog_label_onehot_encoder = OneHotEncoder(sparse=False).fit(encoded_dog_labels.reshape(-1, 1))
onehot_encoded_dog_labels = dog_label_onehot_encoder.transform(encoded_dog_labels.reshape(-1, 1))


# In[5]:


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


# In[6]:


tensors = paths_to_tensor(file_paths).astype('float32')
preprocessed_input = preprocess_input(tensors)
preprocessed_tensors = InceptionV3(weights='imagenet', include_top=False).predict(preprocessed_input, batch_size=32)
print("InceptionV3 shape", preprocessed_tensors.shape[1:])


# In[9]:


def input_branch(input_shape=None):
    
    size = int(input_shape[2] / 4)
    
    branch_input = Input(shape=input_shape)
    branch = GlobalAveragePooling2D()(branch_input)
    branch = Dense(size, use_bias=False, kernel_initializer='uniform')(branch)
    branch = BatchNormalization()(branch)
    branch = Activation("relu")(branch)
    return branch, branch_input

inception_v3_branch, inception_v3_input = input_branch(input_shape=(8, 8, 2048))
net = Dropout(0.3)(inception_v3_branch)
net = Dense(512, use_bias=False, kernel_initializer='uniform')(net)
net = BatchNormalization()(net)
net = Activation("relu")(net)
net = Dropout(0.3)(net)
net = Dense(120, kernel_initializer='uniform', activation="softmax")(net)

dog_breed_model = Model(inputs=[inception_v3_input], outputs=[net])
dog_breed_model.summary()


# In[10]:


dog_breed_model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
dog_breed_model.fit([preprocessed_tensors], onehot_encoded_dog_labels,
          epochs=10, batch_size=4, verbose=1)
dog_breed_model.save_weights('dogbreed_model.hdf5')


# In[11]:


# Save the label encoder
with open('dog_label_encoder.pickle', 'wb') as handle:
    pickle.dump(dog_label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

