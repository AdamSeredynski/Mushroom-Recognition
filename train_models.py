import numpy as np 
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from project_shared import IMAGE_SIZE, load_classes

def load_images():        
    images = []
    labels = []
    
    for class_name in class_list:
        class_path = ('data_model/data/' + class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = load_img(image_path, target_size=IMAGE_SIZE)
            image = img_to_array(image)
            images.append(image)
            labels.append(class_list.index(class_name))
            
    images = np.array(images)
    labels = np.array(labels)
    labels = tf.keras.utils.to_categorical(labels)
    return images, labels

def make_VGG16_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE)+(3,))

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)

    predictions = layers.Dense(len(class_list), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in model.layers[:-4]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def model_evaluations(model):
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    tp = np.sum((y_pred_classes == 1) & (y_true == 1))
    fp = np.sum((y_pred_classes == 1) & (y_true != 1))
    fn = np.sum((y_pred_classes != 1) & (y_true == 1))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    print('True Positives (TP):', tp)
    print('False Positives (FP):', fp)
    print('False Negatives (FN):', fn)
    print('Precision:', round(precision,2))
    print('Recall:', round(recall,2))
    print('F1-score:', round(f1,2))

    model.summary()

class_list = load_classes()
images, labels = load_images()

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=314)

# enabling GPU for processing
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')

# making, training and saving VGG16 model
modelVGG16 = make_VGG16_model()
modelVGG16.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
model_evaluations(modelVGG16)
modelVGG16.save('modelVGG16.h5')


