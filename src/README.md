# Source Code

Here follows a brief description of how the source code is organised and what are the different steps of the processing.

## Step 1: Image and Segmentation Preprocessing

The pipeline requires paired CT image and segmentation both saved in native resolution in separate (`.nrrd`) files. For DICOM to NRRD conversion steps, the (`dcm_to_nrrd`) script in (`/utils`) may be used.

1) Image and segmentation preprocessing 

Data preprocessing (`preprocessingroi_nrrd.py`); includes interpolation to 1x1x3 mm spacing, 10 mm circumferential ROI dilation, cropping and centering the lymph node region of interest

2) Test for ENE and Nodal Metastasis with DualNet

1. Test script (`test_ene.py`)
* Specifies default model and hyperparameters
* Loads data (`data_ene_test.py`); includes intensity normalization; generation of "box input" (118x118x32) and "zoomed-in", size invariant input (32x32x32)
* Runs model inference and saves results to csv