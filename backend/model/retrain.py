import tensorflow as tf 
from tensorflow.keras.models import model_from_json
# from tensorflow.keras.preprocessing import image_dataset_from_directory
# from modelstruct import build_model
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


with open (os.path.join(BASE_DIR, "model.json"),"r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)

model.load_weights(os.path.join(BASE_DIR, "model_weights.h5"))

#adam is popular optimaztion algo used which adjust the learning rate dynamizally  
model.compile(optimizer =Adam(learning_rate=1e-5), loss ="binary_crossentropy",metrics =["accuracy"])
print("model loaded successfully")

dataset2 = "/home/codenia/PROJECTS/Tuberculosis-TB-Analyzer-master/new_app/TB_Chest_Radiography_Database"
Img_Size =(96,96)

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.2,  
    zoom_range=0.2,  
    rescale=1./255,
    validation_split=0.3


)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

train_dataset = datagen.flow_from_directory(
    dataset2,
    subset ="training",
    seed =42,
    target_size = Img_Size,
    batch_size = 32,
    class_mode = "binary",
    shuffle = True
)

val_dataset = val_datagen.flow_from_directory(
    dataset2,
    subset ="validation",
    seed =42,
    target_size = Img_Size,
    batch_size = 32,
    class_mode = "binary",
    shuffle = False

)



retraining = model.fit(train_dataset,epochs=5 ,validation_data =val_dataset )
print("model has been retrained")

model.save_weights(os.path.join(BASE_DIR, "model_weights.h5"))
print("model weights saved successfully")


