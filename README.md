<br/> 

<p align = "center"> <img src="./logo.png" /> 

## **Repository description**

This repository includes all the content developed for the capstone project of the NeueFische Data Science Bootcamp (hh-ds-23-2).

See more details of the project on __[our Capstone Presentation](./Building_Damage_Classification.pdf)__.
## **The project**

**Title**: Building Damage Classification

**Description**: Our project aims to explore deep learning techniques to assess damage to buildings after a natural disaster using satellite imagery.

**Dataset**: The dataset used in this project corresponds to a subset of the xView2 challenge dataset (png images + json labels), accessible on the __[xView2 website](https://xview2.org/dataset)__.  It contains pre- and post-disaster satellite images, polygons of buildings and damage level of each of these buildings.

### ---------------------------------------------------------------
## **Files in this repository**
### **Models**

### 1. stepwise_model_segmentation.ipynb
Uses the output tensors in npz format (pre-disaster) from "get_tensors_pre_post.ipynb" to perform segmentation (identifying buildings).

### 2. stepwise_model_classification.ipynb
Uses the output tensors in npz format (post-disaster) from "get_tensors_pre_post.ipynb" to perform damage classification.

A **python script** to run segformer semantic segmentation pre-trained model is in **./models/segformer_classification.py**. 

**Usage:**
```
segformer_classification.py [--batchsize BATCHSIZE] [--epoch EPOCH] [--augm]  npz_post_file  model_name

positional arguments:

npz_post_file        tensors from post disaster images and masks created with "get_tensors_pre_post.ipynb"   

model_name           identifier of running configuration. Used to save model output.

optional arguments:

--epoch EPOCH, -e EPOCH
        Number of sweeps over the dataset to train

--batchsize BATCHSIZE, -b BATCHSIZE
        Number of images in each mini-batch

--augm , -a
        If chosen applies augmentation techniques (edit script to chose which augmentation transformations are applied)
```

### **Tools**
### 3. EDA_dataset.ipynb 
Have a first look at the dataset. This notebook contains useful functions to extract information contained in JSON format.
### 4. create_png_segmentation_mask.ipynb 
Notebook to create png masks showing buildings (white) and background (black).
### 5. create_png_classification_mask.ipynb
Notebook to create png masks showing damage to buildings in the colours chosen for this project assigned to the 5 different categories (see nb for full details on the damage scale).
### 6. EDA_from_tensors.ipynb 
Give a look to your dataset, plot particular image-mask pairs and visualize data distributions using npz files created in "get_tensors_pre_post.ipynb" notebook.
### 7. create_subset.ipynb
Create a subset from xView2 challenge dataset based on selected maximum size (in gb).
This notebook creates tensors stores in npy format, to use in my of the notebooks of this repository. It requires a folder containing the data divided by disaster. In each disaster folder there are the directories: images (png files) and labels (json files).  
### 8. get_tensors_pre_post.ipynb
This notebook creates pre- and post-tensors containing images, masks and other information relevant to the disaster obtained from JSON files.
### 9. evaluation_stepwise_classification.ipynb
Notebook to get an estimation of the performance of SegFormer on the damage classification task. It includes a pixelwise F1 score estimation by class and displays image-mask-prediction figures. 
### 10. inference.ipynb
This notebook allows you to see post-disaster images and predicted damage masks for any new dataset. It uses a saved version of the model created by "stepwise_model_classification.ipynb", in this case already trained on our own dataset (a subset of the xView2 dataset for challenge).
## Evaluation
The "evaluation_notebooks" directory in notebooks folder contains several evaluation_stepwise_classification.ipynb notebooks separated by disaster.

## Requirements

### Setup

Use the requirements file in this repo to create a new environment.

```BASH
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` file contains the libraries needed for running all jupyter notebooks contained in this repository.



