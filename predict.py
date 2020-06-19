import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default=0,
    help="Model to use when making inference",
    required=False, type=int)
ap.add_argument("-i", "--image", required=True,
    help="Provide a path to image for inference")
args = vars( ap.parse_args() )

# Model Location
path = "./Trained Models/"

# Create an empty list to store models
models = []

# Add models to models list
for i in os.listdir(path):
    models.append(i)

print("Loading model {}".format( models[args["model"]] ) )
start_time = time.time()
infer_model = tf.keras.models.load_model( path + models[args["model"]] )
print("Model loaded in {} seconds".format( time.time() - start_time ))
img = image.load_img( args["image"], color_mode="grayscale", target_size=(250, 250) )
img_array = image.img_to_array( img )
expand_img = np.expand_dims(img_array, axis=0)
flatten_img = np.vstack([expand_img])
classes = infer_model.predict(flatten_img, batch_size=10)

if classes[0][0] == 0:
    print("Patient has pneumonia")
elif classes[0][0] == 1:
    print("Patient does not have pneumonia")
else:
    print("Prediction is inconclusive")
    print(classes[0])