import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


def build_model():
    model = Sequential()

    # First  Block
    model.add(Conv2D(32, (3,3), activation='relu',padding="same", input_shape=(96, 96, 3)))
    model.add(Conv2D(32, (3,3), activation='relu',padding="same"))
    model.add(Conv2D(32, (3,3), activation='relu',padding="same"))
    model.add(MaxPooling2D(pool_size=(2,2))) #  used this for downsampeling the feature map 
    model.add(Dropout(0.3)) # used to remove the overfittinf condition


    # Second  Block
    model.add(Conv2D(64, (3,3), activation='relu',padding="same"))
    model.add(Conv2D(64, (3,3), activation='relu',padding="same"))
    model.add(Conv2D(64, (3,3), activation='relu',padding="same"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))


    # Third  Block
    model.add(Conv2D(128, (3,3), activation='relu',padding="same"))
    model.add(Conv2D(128, (3,3), activation='relu',padding="same"))
    model.add(Conv2D(128, (3,3), activation='relu',padding="same"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))


    # fourth Block
    model.add(Conv2D(256, (3,3), activation='relu',padding="same"))
    model.add(Conv2D(256, (3,3), activation='relu',padding="same"))
    model.add(Conv2D(256, (3,3), activation='relu',padding="same"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    #fully connected layer 
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    # now i am compiliing the model 
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # i need to keep the learning rate upto 0.00001 so that the updates can be slow and stable ..
                  loss='binary_crossentropy',
                  metrics=['accuracy'])#track accuracy during training ...crct_pred/ttl_pred
   
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()   


# output of this code 
# Total params: 4316162 (16.46 MB)
# Trainable params: 4316162 (16.46 MB)
# Non-trainable params: 0 (0.00 Byte)
# ____________________________________________