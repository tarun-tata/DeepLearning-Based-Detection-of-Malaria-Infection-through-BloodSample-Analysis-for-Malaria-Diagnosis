import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")

# Download The Dataset from 
#https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np
from glob import glob
import json
from tensorflow.keras.callbacks import Callback

# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'Dataset/Train'
valid_path = 'Dataset/Test'

target_status_file = 'training_status.json'

class EpochStatus(Callback):
    def __init__(self, stage):
        super().__init__()
        self.stage = stage

    def _write_status(self, status_data):
        with open(target_status_file, 'w') as f:
            json.dump(status_data, f)

    def on_train_begin(self, logs=None):
        self._write_status({
            'status': 'training',
            'stage': self.stage,
            'progress': 'started',
            'epoch': 0,
            'total_epochs': self.params.get('epochs', 0)
        })

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        status_data = {
            'status': 'training',
            'stage': self.stage,
            'epoch': epoch + 1,
            'total_epochs': self.params.get('epochs', 0),
            'loss': float(logs.get('loss', 0.0)) if logs.get('loss') is not None else None,
            'accuracy': float(logs.get('accuracy', logs.get('acc', 0.0))) if logs.get('accuracy', logs.get('acc')) is not None else None,
            'val_loss': float(logs.get('val_loss', 0.0)) if logs.get('val_loss') is not None else None,
            'val_accuracy': float(logs.get('val_accuracy', logs.get('val_acc', 0.0))) if logs.get('val_accuracy', logs.get('val_acc')) is not None else None,
            'progress': f'Epoch {epoch + 1}/{self.params.get("epochs", 0)}'
        }
        self._write_status(status_data)

    def on_train_end(self, logs=None):
        self._write_status({
            'status': 'training',
            'stage': self.stage,
            'progress': 'waiting for next stage',
            'epoch': self.params.get('epochs', 0),
            'total_epochs': self.params.get('epochs', 0)
        })

# add preprocessing layer to the front of VGG
vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
  layer.trainable = False
  

  
  # useful for getting number of classes
folders = glob('Dataset/Train/*')
  

# our layers - you can add more if you want
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.15,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   brightness_range=[0.8,1.2],
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

test_set = test_datagen.flow_from_directory(valid_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

with open(target_status_file, 'w') as f:
    json.dump({"status": "training", "progress": "started", "stage": "phase1", "epoch": 0, "total_epochs": 10}, f)

try:
    with open('class_indices.json', 'w') as f:
        json.dump(training_set.class_indices, f)

    checkpoint = ModelCheckpoint('model_vgg19_best.keras', monitor='val_accuracy', save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    counts = np.bincount(training_set.classes)
    total = float(len(training_set.classes))
    num_classes = len(counts)
    class_weights = {i: total / (num_classes * counts[i]) for i in range(num_classes)}

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])
    status_callback = EpochStatus(stage='phase1')
    r1 = model.fit(
        training_set,
        validation_data=test_set,
        epochs=10,
        steps_per_epoch=len(training_set),
        validation_steps=len(test_set),
        callbacks=[checkpoint, reduce_lr, early_stop, status_callback],
        class_weight=class_weights
    )

    for layer in vgg.layers:
        if 'block5' in layer.name:
            layer.trainable = True

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-5), metrics=['accuracy'])
    fine_tune_callback = EpochStatus(stage='fine_tune')
    r2 = model.fit(
        training_set,
        validation_data=test_set,
        epochs=15,
        steps_per_epoch=len(training_set),
        validation_steps=len(test_set),
        callbacks=[checkpoint, reduce_lr, early_stop, fine_tune_callback],
        class_weight=class_weights
    )

    model.save('model_vgg19_best.keras')
    
    with open('training_status.json', 'w') as f:
        json.dump({"status": "completed", "progress": "finished", "timestamp": str(np.datetime64('now'))}, f)
        
except Exception as e:
    with open('training_status.json', 'w') as f:
        json.dump({"status": "error", "message": str(e)}, f)
    raise e

