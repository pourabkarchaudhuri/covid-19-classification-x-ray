# import tensorflow.keras
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import tensorflow as tf


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

MODEL_PATH = os.path.join(os.getcwd(), 'converted_keras', 'keras_model.h5')
# Load the model
model = load_model(MODEL_PATH)
graph = tf.get_default_graph()
# # Create the array of the right shape to feed into the keras model
# # The 'length' or number of images you can put into the array is
# # determined by the first position in the shape tuple, in this case 1.
# data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# # Replace this with the path to your image
# image = Image.open(os.path.join(os.getcwd(), 'uploads', '1584427208897_1-s2.0-S0140673620303706-fx1_lrg.jpg'))

# #resize the image to a 224x224 with the same strategy as in TM2:
# #resizing the image to be at least 224x224 and then cropping from the center
# size = (224, 224)
# image = ImageOps.fit(image, size, Image.ANTIALIAS)

# #turn the image into a numpy array
# image_array = np.asarray(image)

# # display the resized image
# # image.show()

# # Normalize the image
# normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# # Load the image into the array
# data[0] = normalized_image_array

# # run the inference
# prediction = model.predict(data)
# print("Prediction")
# print(prediction)

def classify(image_path):
    global graph
    with graph.as_default():
# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
        print("Classifying....")
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        # Replace this with the path to your image
        print("Image path {}".format(image_path))
        image = Image.open(image_path)
        # image = Image.open(os.path.join(os.getcwd(), 'uploads', '1584427208897_1-s2.0-S0140673620303706-fx1_lrg.jpg'))

        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        #turn the image into a numpy array
        image_array = np.asarray(image)
        print("As array...")
        # display the resized image
        # image.show()

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        print("Normalized...")
        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        print("Running innference")
        prediction = model.predict(data)
        print("Prediction....")
        print(prediction)
        return prediction
