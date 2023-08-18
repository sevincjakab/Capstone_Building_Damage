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

### **Tools**
### 3. EDA_dataset.ipynb 
Have a first look at the dataset. This notebook contains useful functions to extract information contained in JSON format.
### 4. create_png_segmentation_mask.ipynb 
Notebook to create png masks showing buildings (white) and background (black).
### 5. create_png_classification_mask.ipynb
Notebook to create png masks showing damage to buildings in the colours chosen for this project assigned to the 5 different categories (see nb for full details on the damage scale).



### 6. EDA_from_tensors.ipynb 
(this will show examples from Nt #6)
### 7. subsetting_by_GB.ipynb
### 8. get_tensors_pre_post.ipynb

Requires A FOLDER THAT CONTAINS THE DATA DIVIDED BY DISASTERS, IN EACH DISASTER FOLDER THERE ARE "IMAGES" (PNG FILES) AND "LABELS" FOLDERS (JSON FILES).
### 9. backup_functions.ipynb

## Inferences

### 10. inference_stepwise_classification.ipynb
This will load the model that is already trained on our own dataset (a subset of the xView2 dataset for challenge) and make predictions. It will calculate pixelwise F1 score. 



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



