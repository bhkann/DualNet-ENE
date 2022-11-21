## Preprocessing Code for Deep Learning Head and Neck Project (c) Written by Benjamin H. Kann, 8/16/2017

## Open python
#python3
## Import Python Libraries
#import sys
#print(sys.version)
#import csv
import os
import sys
import math
import re
import shutil
import pandas as pd
import numpy as np
import pydicom
import matplotlib
from matplotlib.path import Path
from matplotlib import colors
import matplotlib.pyplot as plt
import cv2
import scipy
from scipy.spatial import ConvexHull
from scipy import ndimage
import pylab
import plotly
import PIL
from PIL import ImageDraw
import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
import skimage
from skimage import measure, morphology
from skimage.transform import resize, rescale
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.ndimage import binary_closing, grey_closing, binary_dilation
import SimpleITK as sitk

sys.path.append('/home/bhkann/git-repositories/hn-petct-net/data-utils')
from dcm_to_nrrd import dcm_to_nrrd
from interpolate import interpolate
from crop_roi import crop_roi_ene
from util import get_arr_from_nrrd, get_bbox, bbox2_3D, generate_sitk_obj_from_npy_array


##FUNCTION DEFINITION to read through directory list to pull patient list
def load_pt(path):  
    for dirName, subdirList, fileList in os.walk(path):
        if dirName == path:
            lstID = subdirList
        #for filename in fileList:
            #if "DIR" in filename or filename.startswith('.'): continue  # check whether the file is DICOM
            #print(filename)
            #if type(ds) is 'dicom.dataset.FileDataset':
            #lstFilesDCM.append(os.path.join(dirName,filename))
            #lstFolderPt.append((dirName,subdirname))
            
    return lstID #, lstFilesDCM

## FUNCTION DEFINITION to Read through File list to pull DICOMS for Selected patient ##
def load_scan(path):  
    for dirName, subdirList, fileList in os.walk(path):
        for filename in fileList:
            if "DIR" in filename or filename.startswith('.'): continue  # check whether the file's DICOM
            #if type(ds) is 'dicom.dataset.FileDataset':
            lstFilesDCM.append(os.path.join(dirName,filename))
    return lstFilesDCM

def bbox2_3D(img):
    
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))
    
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    
    rmin=int(rmin)
    rmax=int(rmax)
    cmin=int(cmin)
    cmax=int(cmax)
    zmin=int(zmin)
    zmax=int(zmax)
    
    return rmin, rmax, cmin, cmax, zmin, zmax

def clear_duplicates(dicom_data):
    take_id = ''
    series_ids = {}

    for dicom_slice in dicom_data:
        series_id = dicom_slice.SeriesInstanceUID
        if series_id not in series_ids.keys():
            series_ids[series_id] = 1
        else:
            series_ids[series_id] += 1
    max_len = 0
    for series_id in series_ids.keys():
        if series_ids[series_id] > max_len:
            take_id = series_id
            max_len = series_ids[series_id]
    dicom_data_new = []
    for dicom_slice in dicom_data:
        series_id = dicom_slice.SeriesInstanceUID
        if series_id == take_id:
            dicom_data_new.append(dicom_slice)
    return dicom_data_new
##############################################################################################################################################################
############################## MAIN CODE #####################################################################################################
##############################################################################################################################################################
### Generate Patient Unique ID list and list of DICOM files
PathPatient = "/media/bhkann/HN_RES1/HN_DL/E3311_DL/E3311_DICOM_DL/"
lstID = []  # create an empty list

lstID = load_pt(PathPatient)  #Run function to pull ID list
print(lstID)
print("There are", len(lstID), "scans imported.")

NORMALIZE = True  #Set "True" if want to normalize/zero-center each ROI individually
GLOBALNORM = False # Set True if want to do global normalization on whole data set
DOWNSAMPLE = False
IMAGECHECK = False

###### READ IN LABELING DATA (HN_LN_Label.xlsx) #####
labels_df = pd.read_excel('/media/bhkann/HN_RES1/HN_DL/E3311_DL/E3311_LN_Label.xls') #, index_col=0)
labels_df.head()

labels_df.LN_ID = labels_df['LN_ID'].str.replace('_','')

#ece_dict = {}
#ece_dict = dict(zip(labels_df.LN_ID, labels_df.ECE))
    
#positive_dict = {}
#positive_dict = dict(zip(labels_df.LN_ID, labels_df.Positive))

lncat_dict = {}
lncat_dict = dict(zip(labels_df.LN_ID, labels_df.LNcat)) #0,1,2

certain_dict = {}
certain_dict = dict(zip(labels_df.LN_ID, labels_df.CertaintyOverall)) #1,2,3

######################################################################

###Master List For All Patients, All ROIs###
MasterROIList = []
MasterPtROIList = []
inputDLList = []
inputLabelList = []
inputData = []
inputLabelece = []
inputLabelpos = []
inputLabelcat = []
inputMetadata = []
inputROIvolume = []
inputROIdmax = []
inputCertaintydata = []
#MasterROIcount = xyzrange[3]
toobig=0
notzeroglobal=0
pixelsumglobal=0
minglobal=0
maxglobal=0
maxarraysize = (0,0,0)
errors=0
inexactlist=[]

######## Master For-Loop: Cycles through all patient IDs ###################
lstID = sorted(lstID)

try:
    os.mkdir('/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/')
except:
    print("'/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/' directory already exists")

for n in lstID: #lstID: #:lstID: #[89:90]: E3311_33415 DOES NOT WORK(BAD META DATA) #TEST with desired patient ID range len(lstID)
    if n == 'E3311_33415' or n == 'E3311_33466':
        print("skipped E3311_33415 + E3311_33466: bad meta data")
        continue
    str_n = str(n)
    ### Read ROI CSV and convert to pandas dataframe - # rows = # images in file, # columns = 500
    fhand = open('/media/bhkann/HN_RES1/HN_DL/E3311_DL/E3311_roi/' + n + '_roi.csv')
    #fhand = open('/Users/BHKann/Documents/Yale/Yale Research/MachineLearning_HeadNeck/BackupData/HN_roi/10002_roi.csv')
    data = []
    for line in fhand:
        line = line.rstrip('\n')
        words = line.split(',')
        i = 500 - len(words)
        extender = ['&'] * i  ## Pad to give equal dimensions
        words.extend(extender)
        data.append(words)
    
    ## Create pandas dataframe
    roidf = pd.DataFrame(data) 
    #roidf
    #roidf.info()
    
    ####################### Data Organization #####################
    ## Drop unnecessary columns from dataframe
    ## Keep 0 [ImageNo], 7 [RoiName], 16,17,18 [SOP_IDs], 22,23,24 [mmZ, pxX, pyY], 27,28,29 [mmZ, pxX, pyY]...
    
    ## Identify columns of interest
    cols_of_interest = [0,7,16,17,18]
    multiples = list(range(22, 501, 5)) + list(range(23, 501, 5)) + list(range(24, 501, 5)) #v
    multiples.sort()
    cols_of_interest.extend(multiples)
    
    ## Drop columns not of interest
    roidf.drop([col for col in roidf.columns if col not in cols_of_interest], axis=1, inplace=True)
    
    #Rename DF headers, Remove 1st row
    header = roidf.iloc[0]
    roidf.rename(columns = header)
    roidf = roidf[1:]
    
    ## Numpy Array conversion [Imagenum,roi_name,SOP1,SOP2,SOP3, (mmZ,pX,pY)...(mmZ,pX,pY)]
    roiarray = np.asarray(roidf)
    roiarray.shape
    type(roiarray)
    
    imgnumset=roiarray[:,0:2]  #Array of ROI's and Image Numbers
    #Convert imageNum to Integer
    #np.set_printoptions(threshold=np.nan)
    
    #Pt ID variable
    ptID = imgnumset[0,1]
    ptID = ptID.replace('"', '')
    ptID = ptID.strip()
    #if ptID[4] == 'b':
    #    ptID = ptID.replace('b','2')
    #ptID = ptID[4:ptID.index("_")]  #changed to 4 for YALE patients; for the YALEb, convert to '2' at beginning
    
    
    #Identify number of ROIs in series
    roi_name=roiarray[:,1]
    unique_roi_name=np.unique(roi_name)
    roi_count=unique_roi_name.shape[0]
    
    ## Restructure Multi-dimensional list by ROI name
    roi_table = []
    roi_table_axis =[]
    for i in range(0, roi_count):
        roi_table = []
        for j in range(0, roiarray.shape[0]):
            k=int(unique_roi_name[i].strip('"')[-1:])   #Finds position of character after the "_"
            q=int(roiarray[j,1].strip('"')[-1:])      
            if k==q:  #Compares kth character to qth character
                #print(roiarray[j,1])
                roilist=roiarray[j].tolist()
                roilist=[x for x in roilist if (x != '&') and (x != '')]
                roi_table.append(roilist)
                #print(len(roi_table))
                #print(roi_table) #table for each structure
        roi_table_axis.append(roi_table)
    
    ## Create Numpy Array for Masking - Also Indexing for Image Number / ROI Name
    mask_list=[]
    for i in range (0, len(roi_table_axis)): #Loop through each ROI
        roi_z=roi_table_axis[i]
        mask_roi=[]
        for j in range (0, len(roi_z)): #Loop through each slice within a single ROI
            roi_xy=roi_z[j]
            roi_x=roi_xy[6::3]  #Pull out X-pixel coordinates for given ROI slice
            roi_y=roi_xy[7::3]  #Pull out Y-pixel coordinates for given ROI slice
            roi_zmm = roi_xy[5::3]
            roi_img=[]
            roi_name=[]
            for k in range (0, len(roi_x)): #Loop to create Numpy array
                roi_img=[roi_xy[0]]*len(roi_x)  ##Create list of Image # Value for given ROI slice coordinates
                roi_name=[roi_xy[1]]*len(roi_x)  ##Create list of ROI name for given ROI slice coordinates
                #print("Length X", len(roi_x))
                #print("Length Y", len(roi_y))
            mask_xy = np.column_stack((roi_x, roi_y, roi_zmm, roi_img, roi_name))  ## Creates Numpy Array
            mask_roi.append(mask_xy)
        mask_list.append(mask_roi)
    
    
    ####################################################################
    ## Convex Hull, Mask Generation, and Dicom File Reading ##
    
    #################################################
    #### MASK BY ROI BY SLICE FOR SINGLE PATIENT ####
    #################################################
    lstFilesDCM = []  # clear the DICOM file list per patient
    dicomseries = []
    PathDicom = "/media/bhkann/HN_RES1/HN_DL/E3311_DL/E3311_DICOM_DL/" + n + "/"  ## Read Dicom Series Files for one patient _FIX
    #+ "/PATIENT_ANONYMOUS/"  ## Read Dicom Series Files for one patient _FIX
    dicomseries = load_scan(PathDicom)  #Call Function to load DICOM files
    print(dicomseries[0])
    #dicomseries = load_scan(PathDicom)  #Call Function to load DICOM files
    #print(dicomseries[0])
    dicomseries = [item for item in dicomseries if '.dcm' not in item]
    # Get ref file - 1st dicom in the series
    refds = pydicom.read_file(dicomseries[0])
    
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(refds.Rows), int(refds.Columns), len(lstFilesDCM))
    
    data_type = 'CT'
    input_list = os.listdir(PathDicom)
    for p in input_list:
        if os.path.isdir(os.path.join(PathDicom,p)):
            input_dir = os.path.join(PathDicom,p) #+ '/'
    print('input dir for dcm_to_nrrd: ',input_dir)
    dataset = 'E3311'
    patient_id = n
    output_dir = '/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/' #
    sitk_obj = dcm_to_nrrd(dataset, patient_id, data_type, input_dir, output_dir, image_format='ct', save=True)
    img_origin = sitk_obj.GetOrigin()
    img_direction = sitk_obj.GetDirection()
    #
    ###### Loop for each ROI within Patient Dicom series ######
    for i in range(0, len(mask_list)):  # For each ROI
        inexact=0
        roivolume = 0
        #ArrayDicom = np.zeros(ConstPixelDims, dtype='int16') #Initialize the ROI-specific array
        #arrayinput = np.zeros(ConstPixelDims)  #initialize the add-on input array
        maskroilist = mask_list[i]
        nametagarray = maskroilist[0]
        pt_roi = nametagarray[0,4]  #Unique Pt_ROI identifier (same as label)
        roinametag = pt_roi.replace('_','') #String without underscore
        roinametag = roinametag.strip('"')  #Creates Patient + ROI unique ID integer
        ####Pull in the appropriate label (ECE, positivity, cat=1,2,3)
        #roilabel_ece = ece_dict[roinametag]
        #roilabel_pos = positive_dict[roinametag]
        #roilabel_cat = lncat_dict[roinametag]
        ############
        #roinumtag = int(roinametag[5]) #Returns last digit (ROI number) ##???FIX to 6
        #str_pt_roinumtag = str(roinametag)
        #roinumtag = str_pt_roinumtag[5]
        mask3d = np.zeros(sitk_obj.GetSize())
        #print("mask3d after reset:", np.where(mask3d!=0))
        xmaxslice_ct = 0 #Initialize the count for max slice range for X in EACH ROI
        xminslice_ct = 9999
        midpointx = 0
        ymaxslice_ct = 0 #Initialize the count for max slice range for Y in EACH ROI
        yminslice_ct = 9999
        midpointy = 0
        midslicez = 0
        #Slice loop
        for j in range(0, len(maskroilist)):  #For each slice within ROI
            maskarray = maskroilist[j]  #Numpy array for each ROI slice
            xymask = maskarray[:,:2].astype('float32') #* ConstPixelSpacing[:2]  #Pull out X,Y coordinates of ROI slice
            xymask = xymask.astype('int32')
            imgnum = int(maskarray[0,3]) #"Image Number" of each ROI slice (imgnum + 1 = DICOM file name number)
            zcoord = float(maskarray[0,2])
            #print('z coord of mask: ',zcoord)
            dcmnum = imgnum + 1 #Dicom filename number in folder
            
            ## ROI Pre-processing to Mask ##
            ## 1st - Convex Hull - Scipy - Using Vertices
            hull = ConvexHull(xymask)
            hullarray=hull.points.astype('int32')
            # hullplt=scipy.spatial.convex_hull_plot_2d(hull2)
            # hullplt.show()
            
            ## Find Min-Max for X and Y ranges for deep learning bounding box
            ## Find X min, max, and range
            xmax = np.amax(hullarray[:,0])
            xmin = np.amin(hullarray[:,0])
            ##Find Y min, max, and range
            ymax = np.amax(hullarray[:,1])
            ymin = np.amin(hullarray[:,1])
            
            ## Store the Max Ranges across each ROI
            if xmax > xmaxslice_ct:
                xmaxslice_ct = xmax
            if xmin < xminslice_ct:
                xminslice_ct = xmin
            if ymax > ymaxslice_ct:
                ymaxslice_ct = ymax
            if ymin < yminslice_ct:
                yminslice_ct = ymin
        
            # 2nd - Create Template Background Array - set to dimensions of DICOM slice #
            img = PIL.Image.fromarray(np.zeros(ConstPixelDims[:2]))  #To insert DICOM X,Y dimensions here (ie. 512 x 512)
            #To insert DICOM X,Y dimensions here (ie. 512 x 512)
            
            #img = Image.fromarray(np.zeros([512,512]))  
            draw = ImageDraw.Draw(img) 
            #### MASK CREATION ####
            draw.polygon([tuple(p) for p in hullarray], fill=255)   #Creates polygon to match vertices, fills inside
            mask1 = np.array(img)  # Converts the Image data to a numpy array.
            mask1[mask1 == 255] = 1 #If you want to swap axes: mask1 = np.swapaxes(mask1,0,1) # Swaps the X,Y axes [Necessary to do, b/c draw.polygon function inverts the X-Y values]
            #mask1.shape
            
            ## NEW: CREATE 3D MASK ## TRACK SLICE #
            slices=[]
            files=[]
            dicomseries = sorted(dicomseries)
            ## clear out the .dcm duplicates
            
            for f, g in zip(dicomseries,dicomseries[1:]):
                slice = pydicom.read_file(f)
                slice2 = pydicom.read_file(g)
                #sbs = slice.SpacingBetweenSlices
                #if float(slice.SliceThickness) == float(sbs):
                #   slices.append(slice)
                #    #slices = [pydicom.read_file(s) for s in slice_list]
                if float(slice.SliceThickness) == np.abs(slice.ImagePositionPatient[2] - slice2.ImagePositionPatient[2]):
                    slice.PatientName = (f[-3:]) ## EXCLUSIVELY FOR E3311 DATA
                    slices.append(slice)
                #elif np.abs(slice.ImagePositionPatient[2] - slice2.ImagePositionPatient[2]) in [1.0,1.5,2.0,2.5,3.0,3.5] and slice.ImageOrientationPatient==[1,0,0,0,1,0]:
                elif slice.ImageOrientationPatient==[1,0,0,0,1,0]:
                    slice.PatientName = (f[-3:])
                    slices.append(slice)
                if g==dicomseries[-1]:
                    slice2.PatientName = (g[-3:])
                    slices.append(slice2)
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
            for s in range(0, len(slices)):
                zslice = float(slices[s].ImagePositionPatient[2])
                #print("file name:",f)
                #print('idx', s)
                #print("filenameidx: ",filenameidx)
                #print(zslice,zcoord)
                #print(round(zslice,1),round(zcoord,1))
                filenameidx = int(str(slices[s].PatientName))
                if zslice == zcoord or round(zslice, 2) == round(zcoord, 2):
                    print('matching z and slice:',zslice,filenameidx)
                    #zindex = len(dicomseries)-1-filenameidx
                    zindex = sitk_obj.GetSize()[2]-filenameidx
                    mask3d[:,:,zindex] = mask1 #mask3d[:,:,dicomseries.index(filenameDCM)] = mask1
        if np.sum(mask3d)==0:
            inexactlist.append(pt_roi)
        elif np.sum(mask3d) > 0:
            print('3d mask sum:', mask3d.sum())
            mask3dt = np.swapaxes(mask3d,0,2).copy()
            mask3dt = np.rollaxis(mask3dt,2,1).copy()
            #mask3dt = np.transpose(mask3d, (2,1,0))
            img_mask3d = sitk.GetImageFromArray(mask3dt)
            img_mask3d.SetSpacing(sitk_obj.GetSpacing())
            img_mask3d.SetDirection(img_direction)
            img_mask3d.SetOrigin(img_origin)
            nrrd_file_path = '/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/mask3d_' + pt_roi.strip('"') + '.nrrd'
            nrrdWriter = sitk.ImageFileWriter()
            nrrdWriter.SetFileName(nrrd_file_path)
            nrrdWriter.SetUseCompression(True)
            nrrdWriter.Execute(img_mask3d)
                
        print("inexact matches: ", inexactlist)

#### READ THE CREATED NRRDS [3D MASK and Dicom Raw]
### Load up saved nrrd masks and images
###Interpolate to 1x1x3; dilate, crop/center
#import re
try:
    os.mkdir('/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/interpolated')
except:
    print("'/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/interpolated' directory already exists")


from glob import glob
## Bring in New Images/Labels (from 3D Slicer annotation)
path_input = '/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd'
path_new_image =  '/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/new_raw_images'
path_new_label = '/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/new_raw_labels'
for file in sorted(os.listdir(path_new_label)):
    #if '33224' in file or '33234' in file:
    scan_name = '_'.join(file.split('_')[0:2])
    node_name = '_'.join(re.split(', |_|-|\.', file)[0:3])
    src_file = os.path.join(path_new_label,file)
    dst_file = os.path.join(path_input,'mask3d_' + node_name + '.nrrd')
    dst_image = os.path.join(path_input,'E3311_' + scan_name + '_CT_raw_raw_raw_xx.nrrd')
    print(src_file,dst_file,dst_image)
    shutil.copy(src_file,dst_file)
    shutil.copy(glob(path_new_image + '/*' + scan_name + '*')[0],dst_image)


## INTERPOLATION ##
spacing = (1,1,3) #(0.75,0.75,2.5)
INTERP_DIR = '/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/interpolated' + ''.join(str(spacing))
try:
    os.mkdir(INTERP_DIR)
except:
    print("directory already exists")

img_spacing = spacing
path_input = '/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd'
dataset = 'E3311'
for file in sorted(os.listdir(path_input)): #LOOP Goes through nrrd raw images etc
    if not file.startswith('.') and 'CT' in file:
        patient_id = '-'.join(file.split('_')[1:3])
        print("patient ID: ", patient_id)
        path_to_nrrd = os.path.join(path_input,file)            
        print("image path: ",path_to_nrrd)   
        data_type = "ct" #ct or pet
        # path_to_nrrd = "/data/output/dataset_124_ct_raw_raw_raw_xx.nrrd"
        if spacing[2] == 9:
            sitk_img = sitk.ReadImage(path_to_nrrd)
            img_spacing = list(sitk_img.GetSpacing())
            ## NEED TO FINISH : NEED TO APPEND THE ACTUAL Z to tHE DEFINED X,Y
            img_spacing = tuple(np.append(list(spacing[:2]),img_spacing[2]))
            INTERP_DIR = '/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/interpolated' + ''.join(str(spacing))
            try:
                os.mkdir(INTERP_DIR)
            except:
                print("directory already exists")
        interpolation_type = "linear" #"linear" for image, nearest neighbor for label
        #spacing = (1,1,3) # x,y,z For HN SEG: 1, 1, 3; for ENE input: 0.75, 0.75, ?unchanged?2.5?
        return_type = "numpy_array"
        #output_dir = "/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/interpolated"
        interpolate(dataset, patient_id, data_type, path_to_nrrd, interpolation_type, img_spacing, return_type, INTERP_DIR)
        print("image interpolated")
    if not file.startswith('.') and 'mask3d' in file:
        patient_id = '-'.join(file.split('_')[1:3])
        roi_id = patient_id + '_' + file.replace('.','_').split('_')[3]
        print("patient ID: ", patient_id)
        path_to_nrrd = os.path.join(path_input,file)            
        print("image path: ",path_to_nrrd)   
        data_type = "ct" #ct or pet
        # path_to_nrrd = "/data/output/dataset_124_ct_raw_raw_raw_xx.nrrd"
        if spacing[2] == 9:
            sitk_img = sitk.ReadImage(path_to_nrrd)
            img_spacing = list(sitk_img.GetSpacing())
            ## NEED TO FINISH : NEED TO APPEND THE ACTUAL Z to tHE DEFINED X,Y
            img_spacing = tuple(np.append(list(spacing[:2]),img_spacing[2]))
            INTERP_DIR = '/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/interpolated' + ''.join(str(spacing))
            try:
                os.mkdir(INTERP_DIR)
            except:
                print("directory already exists")
        sitk_label = sitk.ReadImage(path_to_nrrd)
        #spacing_label = sitk_label.GetSpacing()
        interpolation_type = "nearest_neighbor" #"linear" for image, nearest neighbor for label
        #spacing = (1,1,3) # x,y,z For HN SEG: 1, 1, 3; for ENE input: 0.75, 0.75, ?unchanged?
        return_type = "sitk_object"
        #output_dir = "/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/interpolated"
        interpolated_img = interpolate(dataset, patient_id, data_type, path_to_nrrd, interpolation_type, img_spacing, return_type, output_dir="")
        print("label interpolated")
        interpolated_array = sitk.GetArrayFromImage(interpolated_img)
        if np.sum(interpolated_array) > 0:
            dilated_array = binary_dilation(interpolated_array, structure=np.ones((1,5,5),np.int32), iterations=2, mask=None, output=None, border_value=0, origin=0, brute_force=False).astype(interpolated_array.dtype)    
            dilated_img = sitk.GetImageFromArray(dilated_array)
            dilated_img.SetSpacing(interpolated_img.GetSpacing())
            dilated_img.SetDirection(interpolated_img.GetDirection())
            dilated_img.SetOrigin(interpolated_img.GetOrigin())
            nrrd_file_path = INTERP_DIR + '/' + roi_id + '_interpolated.nrrd'
            nrrdWriter = sitk.ImageFileWriter()
            nrrdWriter.SetFileName(nrrd_file_path)
            nrrdWriter.SetUseCompression(True)
            nrrdWriter.Execute(dilated_img)
            print("nrrd saved")


### Center, and Crop to BOX NET
output_folder_image = '/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/image_crop' + ''.join(str(spacing))
output_folder_label = '/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd/label_crop' + ''.join(str(spacing))
try:
    os.mkdir(output_folder_image)
    os.mkdir(output_folder_label)
except:
    print("directory already exists")

path_input = INTERP_DIR 
#PATH0='/media/bhkann/HN_RES1/HN_DL/E3311_DL/ENE_preprocess_nrrd'
dataset = 'E3311'
crop_shape = (118,118,32) # Shape of BoxNet
return_type='numpy_array'
for file in sorted(os.listdir(path_input)):
    if 'ct' not in file:
        label_id = '_'.join(file.split('_')[:2])
        patient_id = label_id.split('_')[0]
        print('label',label_id, 'patient', patient_id)
        path_to_label_nrrd = os.path.join(path_input,file)
        path_to_image_nrrd = os.path.join(path_input,dataset + '_' + patient_id + '_ct_interpolated_raw_raw_xx.nrrd')
        #output_folder_image = os.path.join(PATH0,'image_crop')
        #output_folder_label = os.path.join(PATH0,'label_crop')
        image_obj, label_obj = crop_roi_ene(dataset,
                                        label_id,
                                        label_id,
                                        path_to_image_nrrd, 
                                        path_to_label_nrrd, 
                                        crop_shape,
                                        return_type, 
                                        output_folder_image, 
                                        output_folder_label)



### Renaming masks for radiology review
pre = "label-"
folder_rad = "/media/bhkann/HN_RES1/HN_DL/E3311_DL/E3311_analysis/Rads_Review_Study"
os.chdir(folder_rad)
for f in os.listdir():
    if "mask3d" in f and "label" not in f:
        print(f)
        os.rename(f, pre + f)