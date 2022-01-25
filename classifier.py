import os
import re
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,\
     Dropout,Flatten,Dense,Activation,\
     BatchNormalization


img_height = 28
img_width = 28
batch_size = 2
image_channels=3

model = keras.Sequential(
    [
        layers.Input((28, 28, 1)),
        layers.Conv2D(16, 3, padding="same"),
        layers.Conv2D(32, 3, padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(10),
    ]
)

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    "D:\VSC\CNN\dogs_cats\dataset",
    labels="inferred",
    label_mode="int",  
    
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(img_height, img_width),  
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="training",

)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    "D:\VSC\CNN\dogs_cats\dataset",
    labels="inferred",
    label_mode="int",  
    
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(img_height, img_width),  
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="validation",

)
num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join("D:\VSC\CNN\kagglecatsanddogs_3367a\PetImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Deleting corrupted images
            os.remove(fpath)

print("Deleted %d images" % num_skipped)

def augment(x, y):
    image = tf.image.random_brightness(x, max_delta=0.05)
    return image, y

ds_train = ds_train.map(augment)

for epochs in range(10):
     for x, y in ds_train:
         
         pass

model=Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(img_height,img_width,image_channels)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3),activation='relu'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,(3,3),activation='relu'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
    metrics=["accuracy"],
)
model.fit(ds_train, epochs=10, verbose=2)

test_loss, test_acc = model.evaluate(ds_validation, verbose=2)

print('\nTest accuracy:', test_acc)

