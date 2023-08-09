import os
#import cv2
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
#import matplotlib.pyplot as plt
#import matplotlib as mpl

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend
from transformers import TFSegformerForSemanticSegmentation

from IPython.display import clear_output
from tensorflow.keras.callbacks import History



# Augmentation functions

# adjust brightness of image
# don't alter in mask
def brightness(img, mask):
    img = tf.image.adjust_brightness(img, 0.1)
    return img, mask

# flip both image and mask identically
def flip_hori(img, mask):
    img = tf.image.flip_left_right(img)
    mask = tf.image.flip_left_right(mask)
    return img, mask

# flip both image and mask identically
def flip_vert(img, mask):
    img = tf.image.flip_up_down(img)
    mask = tf.image.flip_up_down(mask)
    return img, mask

# rotate both image and mask identically
def rotate(img, mask):
    img = tf.image.rot90(img)
    mask = tf.image.rot90(mask)
    return img, mask

#apply only to the image
def gamma(img, mask, gamma=0.5):
    img = tf.image.adjust_gamma(img, gamma)
    return img, mask
# --------------------------------------------------------------------------
def map_fn(image, mask):
    # Assign names to the elements in the dataset
    return {"image": image, "segmentation_mask": mask}

# --------------------------------------------------------------------------
def normalize(input_image, input_mask):
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    input_image = (input_image - mean) / tf.maximum(std, backend.epsilon())
    #input_mask -= 1
    return input_image, input_mask


def load_image(datapoint):
    '''
    '''
    image_size = 512
    

    input_image = tf.image.resize(datapoint["image"], (image_size, image_size))
    input_mask = tf.image.resize(
        datapoint["segmentation_mask"],
        (image_size, image_size),
        method="bilinear",
    )
    
    input_image, input_mask = normalize(input_image, input_mask)
    input_image = tf.transpose(input_image, (2, 0, 1))
    return {"pixel_values": input_image, "labels": tf.squeeze(input_mask)} 
# --------------------------------------------------------------------------

def segformer_classification():
    parser = argparse.ArgumentParser()

    parser.add_argument('npz_file', help='Path (file included) to npz with tensors post-image/post-damage-labels are stored')
    parser.add_argument('model_name', help='name of folder where the model will be saved')

    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Number of images in each mini-batch')
    parser.add_argument('--augm', '-a',action='store_true', 
                        help='applies augmentation functions to the train set')
    args = parser.parse_args()

    RANDOM_SEED=0
    # load tensors (post disaster)
    loaded_arrays_post = np.load(args.npz_file)
    images_post = loaded_arrays_post['images']
    masks_post = loaded_arrays_post['masks']

    X = images_post
    y = masks_post
    # splitting train and validation sets
    train_X, val_X,train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    # create tensor datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X,train_y))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_X,val_y))
    
    # perform augmentation on train data only
    if args.augm:
        #a = train_dataset.map(brightness)
        b = train_dataset.map(gamma)
        #c = train_dataset.map(hue)
        #d = train_dataset.map(crop)
        e = train_dataset.map(flip_hori)
        f = train_dataset.map(flip_vert)
        g = train_dataset.map(rotate)

        # concatenate every new augmented sets
        #train_dataset = train_dataset.concatenate(a)
        train_dataset = train_dataset.concatenate(b)
        #train_dataset = train_dataset.concatenate(c)
        #train_dataset = train_dataset.concatenate(d)
        train_dataset = train_dataset.concatenate(e)
        train_dataset = train_dataset.concatenate(f)
        train_dataset = train_dataset.concatenate(g)
    

    named_dataset_train = train_dataset.map(map_fn)
    named_dataset_val = val_dataset.map(map_fn)
    
    # parameters needed by the model to pre-process images and labels
    
    
    # preparing the data for the model
    auto = tf.data.AUTOTUNE
    batch_size = args.batchsize

    train = (
        named_dataset_train
        .cache()
        .shuffle(batch_size * 10)
        .map(load_image, num_parallel_calls=auto)
        .batch(batch_size)
        .prefetch(auto)
    )
    val = (
        named_dataset_val
        .map(load_image, num_parallel_calls=auto)
        .batch(batch_size)
        .prefetch(auto)
    )
    
    # load the model
    model_checkpoint = "nvidia/mit-b5"
    id2label = {0: "background", 1: "no-damage", 2: "minor-damage", 3: "major-damage", 4: "destroyed"}
    label2id = {label: id for id, label in id2label.items()}
    num_labels = len(id2label)
    model = TFSegformerForSemanticSegmentation.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    lr = 0.00006
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer)

    class DisplayCallback(tf.keras.callbacks.Callback):
        def __init__(self, dataset, **kwargs):
            super().__init__(**kwargs)
            self.dataset = dataset

        def on_epoch_end(self, epoch, logs=None):
            clear_output(wait=True)
            print("\nSample Prediction after epoch {}\n".format(epoch + 1))

    model_checkpoint_dir = args.model_name
    history_file_path = os.path.join(model_checkpoint_dir, "history.json")

    # training model
    
    history = model.fit(
    train,
    validation_data=val,
    callbacks=[DisplayCallback(val),History()],
    epochs=args.epoch,
    )
    # Save the model after training
    model.save_pretrained(model_checkpoint_dir)

    


if __name__ == '__main__':
    segformer_classification()
    