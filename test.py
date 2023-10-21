from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
imagename ='artifacts/data_ingestion/pizza_steak/train/steak/3136.jpg'
test_image = image.load_img(imagename, target_size = (224,224,3))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
print(test_image.shape)
m_path='artifacts/training/model.h5'
model = load_model(m_path)
print(model.predict(test_image))