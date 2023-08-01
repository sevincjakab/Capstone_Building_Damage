# 1. To load the dataset: image and mask paths
# 2. Building the TensorFlow Input Data Pipeline using tf.data API

import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf

def load_data(path):
    images = sorted(glob(os.path.join(path, "images/*")))
    masks = sorted(glob(os.path.join(path, "masks/*")))

    return images, masks

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    #x = np.expand_dims(x, axis=-1)
    x = x.astype(np.float32)
    return x

def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()
        print(x,y)

        x = read_image(x)
        y = read_mask(y)

        return x, y

    images, masks = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
    images.set_shape([256, 256, 3])
    masks.set_shape([256, 256, 3])

    return images, masks

def tf_dataset(x, y, batch=4):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset

if __name__ == "__main__":
    path = "/Users/sevincjakab/neuefische_bootcamp/20230717-NewRepo-Capstone-Building_Damage/Capstone_Building_Damage/data/xBD_test_subset_reorganized"
    
    images, masks = load_data(path)
    print(f"Images: {len(images)} - Masks: {len(masks)}")

    # x= read_image(images[0])
    # y= read_image(masks[0])
    # print(x.shape, y.shape)


    dataset = tf_dataset(images, masks)
    for x, y in dataset:
        x = x[0]
        y = y[0]
        print(x,y)
        #print(x.shape, y.shape)

    # If you wan to convert them back to images to 
    # double check if everything worked out
    #     x = x[0] * 255
    #     y = y[0] * 255

    #     x = x.numpy()
    #     y = y.numpy()

    #     cv2.imwrite("image.png", x)

    #     y = np.squeeze(y, axis=-1) # you can skip this 
    #     cv2.imwrite("mask.png", y)

    #     break
