import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from project_shared import IMAGE_SIZE, load_classes

def load_user_images():
    images = []
    image_names = []
    folder_path = 'user_images'
    for p in os.listdir(folder_path):
        image_path = os.path.join(folder_path, p)
        image = load_img(image_path, target_size=IMAGE_SIZE)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        images.append(image)
        image_names.append(p)
    return images, image_names

def predict(model, images, image_names, model_name, classes):
    for i in range(len(images)):
        predictions = model.predict(images[i])
        predicted_class = np.argmax(predictions[0])
        print("Model:", model_name)
        print("File name:", image_names[i])
        print("Predicted class:", classes[predicted_class])
        print("Propability:", predictions[0][predicted_class])
        np.set_printoptions(precision=4, suppress=True)
        highest_indices = np.argsort(predictions[0])[-5:]
        highest_indices = highest_indices[::-1]
        print("\nTop five predictions:")
        for p in highest_indices:
            print(classes[p], ": ", round(predictions[0][p],2))
        print()

classes = load_classes()

model_name = "VGG16"
model = tf.keras.models.load_model("model"+model_name+".h5")

for layer in model.layers:
    print(layer.name, type(layer))

images, image_names = load_user_images()

predict(model, images, image_names, model_name, classes)

