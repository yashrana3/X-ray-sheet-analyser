import tensorflow as tf 
from tensorflow.keras.models import model_from_json
import cv2 
import numpy as np

#first of all we need load the model 
with open("model/model.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)

# now we need to load model weights
model.load_weights("model/model_weights.h5")

# Compiling model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
print("model loaded successfully")

#now to check if our model is working correctly or not we need preprocess it before loading it into the model ....

image = cv2.imread("assets/dataset3/Dataset_of_TuberculosisChestX-rays_Images/Normal/CHNCXR_0004_0.png")
image = cv2.resize(image,(96,96))
image = image.astype("float32")/255.0
#as we now that dl model needs input in batches even for single image 
image = np.expand_dims(image, axis=0) 

prediction = model.predict(image)
threshold = 0.5
predicted_class = 1 if prediction[0][0] > threshold else 0
print(predicted_class)
