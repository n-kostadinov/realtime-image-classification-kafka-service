
# coding: utf-8

# In[1]:


from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers import Input, Dense
from keras.layers.core import Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import img_to_array
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.applications import inception_v3
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
import os.path
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import base64
from PIL import Image
from kafka import KafkaConsumer, KafkaProducer
from io import BytesIO
import json


# In[ ]:


KAFKA_BROKER_ADDRESS='localhost:9092'


# In[2]:


try:
    assert os.path.isfile('dogbreed_model.hdf5') and            os.path.isfile('dogbreed_labels.pickle')
except:
    print("Run the Train_Dog_Breed_Model Script first to train the Dog Breed Classification Model")
    raise


# In[3]:


inception_model = InceptionV3(weights='imagenet', include_top=False)


# In[4]:


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
dog_breed_model.load_weights('dogbreed_model.hdf5')


# In[ ]:


with open("dogbreed_labels.pickle", "rb") as f:
    dogbreed_labels = np.array(pickle.load(f))

def format_percentage(raw_probability):
    return "{0:.2f}%".format(raw_probability * 100)
    
class LabelRecord(object):
    def __init__(self, predictions):
        
        probabilities = np.array(predictions[0])
        top_five_breed_index = np.argsort(probabilities)[::-1][:5]
        
        dog_breed_names = dogbreed_labels[top_five_breed_index]
        
        self.label1 = dog_breed_names[0].upper()
        self.probability1 = format_percentage(probabilities[top_five_breed_index[0]])
        self.label2 = dog_breed_names[1].upper()
        self.probability2 = format_percentage(probabilities[top_five_breed_index[1]])
        self.label3 = dog_breed_names[2].upper()
        self.probability3 = format_percentage(probabilities[top_five_breed_index[2]])
        self.label4 = dog_breed_names[3].upper()
        self.probability4 = format_percentage(probabilities[top_five_breed_index[3]])
        self.label5 = dog_breed_names[4].upper()
        self.probability5 = format_percentage(probabilities[top_five_breed_index[4]])

    def toJSON(self):
        return json.dumps(self, default=lambda obj: obj.__dict__, sort_keys=True, indent=4)


# In[ ]:


# Kafka Service
consumer = KafkaConsumer('classificationimage', group_id='group1',bootstrap_servers=KAFKA_BROKER_ADDRESS)
producer = KafkaProducer(bootstrap_servers=KAFKA_BROKER_ADDRESS)
for message in consumer:
    
    # transform image
    image_data = base64.b64decode(message.value.decode())
    pil_image = Image.open(BytesIO(image_data))
    image_array = img_to_array(pil_image)
    image_batch = np.expand_dims(image_array, axis=0)
    processed_image = preprocess_input(image_batch.copy())
    
    # make predictions
    inception_v3_predictions = inception_model.predict(processed_image)
    predictions = dog_breed_model.predict(inception_v3_predictions)
    
    # transform predictions to json
    label = LabelRecord(predictions)
    label_json = label.toJSON()
    
    # send encoded label
    producer.send('classificationlabel', label_json.encode())

