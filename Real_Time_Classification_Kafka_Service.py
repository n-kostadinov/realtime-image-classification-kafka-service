
# coding: utf-8

# In[1]:


import keras
import numpy as np
from keras.applications import inception_v3
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import base64
from PIL import Image
from kafka import KafkaConsumer, KafkaProducer
from io import BytesIO
import json


# In[ ]:


KAFKA_BROKER_ADDRESS='localhost:9092'


# In[2]:


#Load the Inception_V3 model
inception_model = inception_v3.InceptionV3(weights='imagenet')


# In[ ]:


def format_percentage(raw_probability):
    return "{0:.2f}%".format(raw_probability * 100)
    
class LabelRecord(object):
    def __init__(self, raw_label):
        
        self.label1 = raw_label[0][0][1].upper()
        self.probability1 = format_percentage(raw_label[0][0][2])
        self.label2 = raw_label[0][1][1].upper()
        self.probability2 = format_percentage(raw_label[0][1][2])
        self.label3 = raw_label[0][2][1].upper()
        self.probability3 = format_percentage(raw_label[0][2][2])
        self.label4 = raw_label[0][3][1].upper()
        self.probability4 = format_percentage(raw_label[0][3][2])
        self.label5 = raw_label[0][4][1].upper()
        self.probability5 = format_percentage(raw_label[0][4][2])

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
    processed_image = inception_v3.preprocess_input(image_batch.copy())
    
    # make predictions
    predictions = inception_model.predict(processed_image)
    
    # transform predictions to json
    raw_label = decode_predictions(predictions)
    label = LabelRecord(raw_label)
    label_json = label.toJSON()
    
    # send encoded label
    producer.send('classificationlabel', label_json.encode())

