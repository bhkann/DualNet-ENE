import sys, os, glob
import SimpleITK as sitk
import pydicom
import numpy as np

# Load the scans in given folder path
def load_dicom(slice_list):
    #print(slice_list)
    slices = []
    for s, t in zip(slice_list,slice_list[1:]):
        slice = pydicom.read_file(s)
        slice2 = pydicom.read_file(t)
        #sbs = slice.SpacingBetweenSlices
        #if float(slice.SliceThickness) == float(sbs):
        #   slices.append(slice)
        #    #slices = [pydicom.read_file(s) for s in slice_list]
        if slice.ImageOrientationPatient==[0,1,0,0,0,-1]:
            slice.ImageOrientationPatient=[1,0,0,0,1,0]
            slice2.ImageOrientationPatient=[1,0,0,0,1,0]

        if float(slice.SliceThickness) == np.abs(slice.ImagePositionPatient[2] - slice2.ImagePositionPatient[2]):
            slices.append(slice)
        #elif np.abs(slice.ImagePositionPatient[2] - slice2.ImagePositionPatient[2]) in [1.0,1.5,2.0,2.5,3.0,3.5] and slice.ImageOrientationPatient==[1,0,0,0,1,0]:
        elif slice.ImageOrientationPatient==[1,0,0,0,1,0]:    
            slices.append(slice)
        if t==slice_list[-1]:
            slices.append(slice2)
    try:
        # seriesDesc = slices[0].SeriesDescription
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
            if slice_thickness > 3 or slice_thickness == 0:
                slice_thickness = np.abs(slices[9].ImagePositionPatient[2] - slices[10].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
            if slice_thickness > 3:
                slice_thickness = np.abs(slices[9].SliceLocation - slices[10].SliceLocation)
    except Exception as e:
        print (e)
        print('No position found for image', slice_list[0])
        if 'Sinai'  or 'TCGA' or 'E3311' in slice_list[0]:
            slice_thickness = float(slices[0].SliceThickness)
            print("fall back slice thickness: ", slice_thickness)
        else:
            #slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
            #print("slice thickness: ", slice_thickness)
            return []
    for s in slices:
        s.SliceThickness = slice_thickness
        #print('s.SliceThickness: ', s.SliceThickness, s.SpacingBetweenSlices)
        print('image position:', s.ImagePositionPatient)
    img_spacing = [float(slices[0].PixelSpacing[0]),
    float(slices[0].PixelSpacing[1]), slice_thickness]
    #print('img_spacing: ', img_spacing)
    img_direction = [float(i) for i in slices[0].ImageOrientationPatient] + [0, 0, 1]
    img_origin = slices[0].ImagePositionPatient
    
    return slices, img_spacing, img_direction, img_origin

# Load the PET scans in given folder path
def load_dicom_pet(slice_list):
    slices = [pydicom.read_file(s) for s in slice_list]
    
    try:
        # seriesDesc = slices[0].SeriesDescription
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    except Exception as e:
        print (e)
        #print 'No position found for image', slice_list[0]
        return []
    for s in slices:
        s.SliceThickness = slice_thickness
    
    img_spacing = [float(slices[0].PixelSpacing[0]),
    float(slices[0].PixelSpacing[1]), slice_thickness]
    img_direction = [int(i) for i in slices[0].ImageOrientationPatient] + [0, 0, 1]
    img_origin = slices[0].ImagePositionPatient
    
    return slices, img_spacing, img_direction, img_origin
    
    
def getPixelArray(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    #image[image == -2000] = 0
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)

def run_core(dicom_dir, image_format):
    print ('Processing patient ', dicom_dir)
    #dicomFiles = sorted(glob.glob(dicom_dir + '/*.dcm'))
    dicomFiles = sorted(glob.glob(dicom_dir + '/[!D]*'))
    dicomCheck = list(map(lambda sub:int(''.join([i for i in sub if i.isnumeric()])), dicomFiles)) #Extracts only the numbers to check for duplicates
    if len(dicomCheck) > len(set(dicomCheck)):
        dicomFiles = [item for item in dicomFiles if '.dcm' not in item]
        print("Removing duplicate slices")
    if image_format=='ct':
        slices, img_spacing, img_direction, img_origin = load_dicom(dicomFiles)
        print('img_spacing: ', img_spacing)
    elif image_format=='pet':
        slices, img_spacing, img_direction, img_origin = load_dicom_pet(dicomFiles)
    if 0.0 in img_spacing:
        print ('ERROR - Zero spacing found for patient,', img_spacing)
        return ''
    
    
    if image_format=='ct':
        imgCube = getPixelArray(slices)
    elif image_format=='pet':
        imgCube = getPixelArray_pet(slices, image_format)   
    # Build the SITK nrrd image
    imgSitk = sitk.GetImageFromArray(imgCube)
    imgSitk.SetSpacing(img_spacing)
    imgSitk.SetDirection(img_direction)
    imgSitk.SetOrigin(img_origin)
    
    return imgSitk
    #except:

def dcm_to_nrrd(dataset, patient_id, data_type, input_dir, output_dir, image_format, save=True):
    """
    Converts a stack of slices into a single .nrrd file and saves it.
    Args:
        dataset (str): Name of dataset.
        patient_id (str): Unique patient id.
        data_type (str): Type of data (e.g., ct, pet, mri..)
        input_dir (str): Path to folder containing slices.
        output_dir (str): Path to folder where nrrd will be saved.
        save (bool): If True, the nrrd file is saved
    Returns:
        The sitk image object.
    Raises:
        Exception if an error occurs.
    """
    try:
        nrrd_name = "{}_{}_{}_raw_raw_raw_xx.nrrd".format(dataset, patient_id, data_type)
        nrrd_file_path = os.path.join(output_dir, nrrd_name)
        sitk_object = run_core(input_dir, image_format)
        if save:
            nrrdWriter = sitk.ImageFileWriter()
            nrrdWriter.SetFileName(nrrd_file_path)
            nrrdWriter.SetUseCompression(True)
            nrrdWriter.Execute(sitk_object)
        print ("dataset:{} patient_id:{} done!".format(dataset, patient_id))
        return sitk_object
    except Exception as e:
        print ("dataset:{} patient_id:{} error:{}".format(dataset, patient_id, e))