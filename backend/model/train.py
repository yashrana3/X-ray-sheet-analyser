import numpy as np
import tensorflow as tf 
from modelstruct import build_model
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset = "/home/codenia/PROJECTS/Tuberculosis-TB-Analyzer-master/new_app/TB_Chest_Radiography_Database"
Img_Size = (96, 96) 

# Loading Training Dataset
train_dataset = image_dataset_from_directory(
    dataset,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=Img_Size, 
    batch_size=32,
    label_mode="binary"
)

# Loading Validation Dataset
val_dataset = image_dataset_from_directory(
    dataset,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=Img_Size, 
    batch_size=32,
    label_mode="binary"
)

# Building Model
modelstruct = build_model()

# Training Model
modelstruct.fit(train_dataset, epochs=10, validation_data=val_dataset)

model_json = modelstruct.to_json()
with open(os.path.join(BASE_DIR, "model.json"), "w") as json_file:
    json_file.write(model_json)

# Saving model weights
modelstruct.save_weights(os.path.join(BASE_DIR, "model_weights.h5"))

