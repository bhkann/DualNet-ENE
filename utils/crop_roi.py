import os
import operator
import numpy as np
import SimpleITK as sitk
from util import get_arr_from_nrrd, get_bbox, generate_sitk_obj_from_npy_array
#from scipy.ndimage import sobel, generic_gradient_magnitude
from scipy import ndimage

def pad_helper(center, mid, right_lim, axis):
    """
    Helps in adjustting center points and calculating padding amounts on any axis.
    Args:
        center (int): center index from array_to_crop_from
        mid (int): midpoint of axes of shape to be cropped to
        right_lim (int): right limit of axes of shape to be cropped to, after which padding will be needed
        axis (str): string of which axes "X", "Y, or "Z". For debugging.
    Returns:
        center (int): adjusted center
        pad_l (int): amount of padding needed on the left side of axes
        pad_r (int): amount of padding needed on the right side of axes
    """
    pad_l = 0
    pad_r = 0

    if center < mid:
        pad_l = mid - center
        print ("{} left shift , padding :: {}, center :: {}, mid :: {}".format(axis, pad_l, center, mid))
        center = mid #moves center of the label bbox to the center of the desired cropped shape

    # if we are left padding, update the right_lim
    right_lim = right_lim + pad_l

    if center > right_lim:
        pad_r = center - right_lim
        print ("{} right shift , padding :: {}, center :: {}, right_lim :: {}".format(axis, pad_r, center, right_lim))
        # do not change center here

    return  center, pad_l, pad_r


def crop_and_pad(array_to_crop_from, shape_to_crop_to, center, pad_value):
    """
    Will crop a given size around the center, and pad if needed.
    Args:
        array_to_crop_from: array to crop form.
        shape_to_crop_to (list) shape to save cropped image  (x, y, z)
        center (list) indices of center (x, y, z)
        pad_value
    Returns:
        cropped array
    """
    # constant value halves of requested cropped output
    X_mid = shape_to_crop_to[0]//2 
    Y_mid = shape_to_crop_to[1]//2 
    Z_mid = shape_to_crop_to[2]//2 
    
    # right-side limits based on shape of input
    X_right_lim = array_to_crop_from.shape[2]-X_mid
    Y_right_lim = array_to_crop_from.shape[1]-Y_mid
    Z_right_lim = array_to_crop_from.shape[0]-Z_mid
    print("right lims",X_right_lim,Y_right_lim,Z_right_lim)
    
    # calculate new center and shifts
    X, x_pad_l, x_pad_r = pad_helper(center[0], X_mid, X_right_lim, "X")
    Y, y_pad_l, y_pad_r = pad_helper(center[1], Y_mid, Y_right_lim, "Y")
    Z, z_pad_l, z_pad_r = pad_helper(center[2], Z_mid, Z_right_lim, "Z")
        
    # pad
    array_to_crop_from_padded = np.pad(array_to_crop_from,
    ((z_pad_l, z_pad_r),
    (y_pad_l, y_pad_r),
    (x_pad_l, x_pad_r)), 'constant', constant_values=pad_value)

    # get limits
    Z_start, Z_end = Z-Z_mid, Z+Z_mid
    Y_start, Y_end = Y-Y_mid, Y+Y_mid
    X_start, X_end = X-X_mid, X+X_mid

    return array_to_crop_from_padded[Z_start:Z_end, Y_start:Y_end, X_start:X_end]


def crop_roi(dataset, patient_id, path_to_image_nrrd, path_to_label_nrrd, crop_shape, return_type, output_folder_image, output_folder_label):
    """
    Will crop around the center of bbox of label.
    Args:
        dataset (str): Name of dataset.
        patient_id (str): Unique patient id.
        path_to_image_nrrd (str): Path to image nrrd file.
        path_to_label_nrrd (str): Path to label nrrd file.
        crop_shape (list) shape to save cropped image  (x, y, z)
        return_type (str): Either 'sitk_object' or 'numpy_array'.
        output_folder_image (str) path to folder to save image nrrd
        output_folder_label (str) path to folder to save label nrrd
    Returns:
        Either a sitk image object or a numpy array derived from it (depending on 'return_type') of both image and label.
    Raises:
        Exception if an error occurs.
    """
    try:
        # get image, arr, and spacing (returns Z,X,Y order)
        image_obj, image_arr, image_spacing, image_origin = get_arr_from_nrrd(path_to_image_nrrd, "image")
        label_obj, label_arr, label_spacing, label_origin = get_arr_from_nrrd(path_to_label_nrrd, "label")
        #assert image_arr.shape==label_arr.shape, "image & label shape do not match!"

        # get center. considers all blobs
        bbox = get_bbox(label_arr) ### Compare bbox[6] , bbox[7], bbox[8] to crop_shape - make sure 6,7,8 is smaller than crop_shape
        Z, Y, X = int(bbox[9]), int(bbox[10]), int(bbox[11]) # returns center point of the label array bounding box
        print("Original Centroid: ", X, Y, Z)
        
        #find origin translation from label to image
        print("image origin: ", image_origin, " label origin: ",label_origin)
        origin_dif = tuple(np.subtract(label_origin,image_origin).astype(int))
        print("origin difference: ", origin_dif)
        
        X_shift, Y_shift, Z_shift = tuple(np.add((X,Y,Z),np.divide(origin_dif,(1,1,3)).astype(int))) # 
        print("Centroid shifted: ", X_shift, Y_shift, Z_shift)
        
        image_arr_crop = crop_and_pad(image_arr, crop_shape, (X_shift,Y_shift,Z_shift), -1024)
        label_arr_crop = crop_and_pad(label_arr, crop_shape, (X,Y,Z), 0)
        
        #np.save()
        
        output_path_image = os.path.join(output_folder_image, "{}_{}_image_interpolated_roi_raw_gt.nrrd".format(dataset, patient_id))
        output_path_label = os.path.join(output_folder_label, "{}_{}_label_interpolated_roi_raw_gt.nrrd".format(dataset, patient_id))
        
        # save nrrd
        image_crop_sitk = generate_sitk_obj_from_npy_array(label_obj, image_arr_crop, resize=False, output_dir=output_path_image)
        label_crop_sitk = generate_sitk_obj_from_npy_array(label_obj, label_arr_crop, resize=False, output_dir=output_path_label)

        if return_type == "sitk_object":
            return image_crop_sitk, label_crop_sitk
        elif return_type == "numpy_array":
            return image_arr_crop, label_arr_crop

    except Exception as e:
        print ("Error in {}_{}, {}".format(dataset, patient_id, e))
        
def crop_roi_ene(dataset, patient_id, label_id, path_to_image_nrrd, path_to_label_nrrd, crop_shape, return_type, output_folder_image, output_folder_label):
    """
    Will crop around the center of bbox of label.
    Args:
        dataset (str): Name of dataset.
        patient_id (str): Unique patient id.
        path_to_image_nrrd (str): Path to image nrrd file.
        path_to_label_nrrd (str): Path to label nrrd file.
        crop_shape (list) shape to save cropped image  (x, y, z)
        return_type (str): Either 'sitk_object' or 'numpy_array'.
        output_folder_image (str) path to folder to save image nrrd
        output_folder_label (str) path to folder to save label nrrd
    Returns:
        Either a sitk image object or a numpy array derived from it (depending on 'return_type') of both image and label.
    Raises:
        Exception if an error occurs.
    """
    try:
        # get image, arr, and spacing (returns Z,X,Y order)
        image_obj, image_arr, image_spacing, image_origin = get_arr_from_nrrd(path_to_image_nrrd, "image")
        label_obj, label_arr, label_spacing, label_origin = get_arr_from_nrrd(path_to_label_nrrd, "label")
        #assert image_arr.shape==label_arr.shape, "image & label shape do not match!"

        # get center. considers all blobs
        bbox = get_bbox(label_arr) ### Compare bbox[6] , bbox[7], bbox[8] to crop_shape - make sure 6,7,8 is smaller than crop_shape
        Z, Y, X = int(bbox[9]), int(bbox[10]), int(bbox[11]) # returns center point of the label array bounding box
        print("Original Centroid: ", X, Y, Z)
        
        #find origin translation from label to image
        print("image origin: ", image_origin, " label origin: ",label_origin)
        origin_dif = tuple(np.subtract(label_origin,image_origin).astype(int))
        print("origin difference: ", origin_dif)
        
        X_shift, Y_shift, Z_shift = tuple(np.add((X,Y,Z),np.divide(origin_dif,(1,1,3)).astype(int))) # 
        print("Centroid shifted: ", X_shift, Y_shift, Z_shift)
        
        image_arr_crop = crop_and_pad(image_arr, crop_shape, (X_shift,Y_shift,Z_shift), -1024)
        label_arr_crop = crop_and_pad(label_arr, crop_shape, (X,Y,Z), 0)
        
        #np.save()
        
        output_path_image = os.path.join(output_folder_image, "{}_{}_image_interpolated_roi_raw_gt.nrrd".format(dataset, patient_id))
        output_path_label = os.path.join(output_folder_label, "{}_{}_label_interpolated_roi_raw_gt.nrrd".format(dataset, label_id))
        
        # save nrrd
        image_crop_sitk = generate_sitk_obj_from_npy_array(label_obj, image_arr_crop, resize=False, output_dir=output_path_image)
        label_crop_sitk = generate_sitk_obj_from_npy_array(label_obj, label_arr_crop, resize=False, output_dir=output_path_label)

        if return_type == "sitk_object":
            return image_crop_sitk, label_crop_sitk
        elif return_type == "numpy_array":
            return image_arr_crop, label_arr_crop

    except Exception as e:
        print ("Error in {}_{}, {}".format(dataset, patient_id, e))



def crop_top(dataset, patient_id, path_to_image_nrrd, path_to_label_nrrd, crop_shape, return_type, output_folder_image, output_folder_label):
    """
    Will crop around the center of bbox of label.
    Args:
        dataset (str): Name of dataset.
        patient_id (str): Unique patient id.
        path_to_image_nrrd (str): Path to image nrrd file.
        path_to_label_nrrd (str): Path to label nrrd file.
        crop_shape (list) shape to save cropped image  (x, y, z)
        return_type (str): Either 'sitk_object' or 'numpy_array'.
        output_folder_image (str) path to folder to save image nrrd
        output_folder_label (str) path to folder to save label nrrd
    Returns:
        Either a sitk image object or a numpy array derived from it (depending on 'return_type') of both image and label.
    Raises:
        Exception if an error occurs.
    """
    try:
        # get image, arr, and spacing
        image_obj, image_arr, image_spacing, image_origin = get_arr_from_nrrd(path_to_image_nrrd, "image")
        label_obj, label_arr, label_spacing, label_origin = get_arr_from_nrrd(path_to_label_nrrd, "label")
        #assert image_arr.shape==label_arr.shape, "image & label shape do not match!"
        
        # get center. considers all blobs
        bbox = get_bbox(label_arr) ### Compare bbox[6] , bbox[7], bbox[8] to crop_shape - make sure 6,7,8 is smaller than crop_shape
        Z, Y, X = int(bbox[9]), int(bbox[10]), int(bbox[11]) # returns center point of the label array bounding box
        print("Original Centroid: ", X, Y, Z)
        
        #find origin translation from label to image
        print("image origin: ", image_origin, " label origin: ",label_origin)
        origin_dif = tuple(np.subtract(label_origin,image_origin).astype(int))
        print("origin difference: ", origin_dif)
        
        X_shift, Y_shift, Z_shift = tuple(np.add((X,Y,Z),np.divide(origin_dif,(1,1,3)).astype(int))) # 
        print("Centroid shifted: ", X_shift, Y_shift, Z_shift)
        
        #image_arr_crop = crop_and_pad(image_arr, crop_shape, (X_shift,Y_shift,Z_shift), -1024)
        #label_arr_crop = crop_and_pad(label_arr, crop_shape, (X,Y,Z), 0)
        
        #np.save()
        #generic_gradient_magnitude(a, sobel)      
        ## Return top 25 rows of 3D volume, centered in x-y space / start at anterior (y=0)?
        print("image_arr shape: ", image_arr.shape)
        c,y,x = image_arr.shape
        
        ## Get center of mass to center the crop in Y plane
        mask_arr = np.copy(image_arr) 
        mask_arr[mask_arr > -500] = 1
        mask_arr[mask_arr <= -500] = 0
        mask_arr[mask_arr >= -500] = 1 
        print("mask_arr min and max:", np.amin(mask_arr), np.amax(mask_arr))
        centermass = ndimage.measurements.center_of_mass(mask_arr) # z,x,y   
        cpoint=c-crop_shape[2]//2
        print("cpoint, ", cpoint)
        centermass = ndimage.measurements.center_of_mass(mask_arr[cpoint,:,:])   
        print("center of mass: ", centermass)
        startx = int(centermass[0] - crop_shape[0]//2)
        starty = int(centermass[1] - crop_shape[1]//2)      
        #startx = x//2 - crop_shape[0]//2       
        #starty = y//2 - crop_shape[1]//2
        startz = int(c - crop_shape[2])
        print("start X,Y,Z: ", startx, starty, startz)
        if startz < 0:
            image_arr = np.pad(image_arr,
                ((abs(startz)//2, abs(startz)//2),
                (0, 0),
                (0, 0)), 'constant', constant_values=-1024)
            label_arr = np.pad(label_arr,
                ((abs(startz)//2, abs(startz)//2),
                (0, 0),
                (0, 0)), 'constant', constant_values=0)
            image_arr_crop = image_arr[0:crop_shape[2],starty:starty+crop_shape[1],startx:startx+crop_shape[0]]
            label_arr_crop = label_arr[0:crop_shape[2],starty:starty+crop_shape[1],startx:startx+crop_shape[0]]
        else:
            #image_arr_crop = image_arr[startz:startz+crop_shape[2],starty:starty+crop_shape[1],startx:startx+crop_shape[0]]
            #label_arr_crop = label_arr[startz:startz+crop_shape[2],starty:starty+crop_shape[1],startx:startx+crop_shape[0]]
            image_arr_crop = image_arr[0:crop_shape[2],starty:starty+crop_shape[1],startx:startx+crop_shape[0]]
            label_arr_crop = label_arr[0:crop_shape[2],starty:starty+crop_shape[1],startx:startx+crop_shape[0]]
        print('Returning top rows')
        
        output_path_image = os.path.join(output_folder_image, "{}_{}_image_interpolated_roi_raw_gt.nrrd".format(dataset, patient_id))
        output_path_label = os.path.join(output_folder_label, "{}_{}_label_interpolated_roi_raw_gt.nrrd".format(dataset, patient_id))
        print("Output label path: ", output_path_label)
        
        ## Clean up Label ##
        ## Binarize and Clean up stray pixels
        label_arr_crop = np.rint(label_arr_crop).astype(int)
        print("sum for orig array: ",label_arr_crop.sum())
        sitk_label = sitk.GetImageFromArray(label_arr_crop)
        sitk_label = sitk.BinaryMorphologicalOpening(sitk_label, [1, 1, 0])
        sitk_label = sitk.BinaryMorphologicalClosing(sitk_label, [1, 1, 0])
        label_arr_crop = sitk.GetArrayFromImage(sitk_label)
        print("sum for cleaned up array: ",label_arr_crop.sum())
        
        # save nrrd
        image_crop_sitk = generate_sitk_obj_from_npy_array(label_obj, image_arr_crop, resize=False, output_dir=output_path_image)
        print("Saving image cropped")
        label_crop_sitk = generate_sitk_obj_from_npy_array(label_obj, label_arr_crop, resize=False, output_dir=output_path_label)
        print("Saving label cropped")
        if return_type == "sitk_object":
            return image_crop_sitk, label_crop_sitk
        elif return_type == "numpy_array":
            return image_arr_crop, label_arr_crop

    except Exception as e:
        print ("Error in {}_{}, {}".format(dataset, patient_id, e))
        

def crop_top_image_only(dataset, patient_id, path_to_image_nrrd, crop_shape, return_type, output_folder_image):
    """
    Will center the image and crop top of image after it has been registered.
    Args:
        dataset (str): Name of dataset.
        patient_id (str): Unique patient id.
        path_to_image_nrrd (str): Path to image nrrd file.
        path_to_label_nrrd (str): Path to label nrrd file.
        crop_shape (list) shape to save cropped image  (x, y, z)
        return_type (str): Either 'sitk_object' or 'numpy_array'.
        output_folder_image (str) path to folder to save image nrrd
        output_folder_label (str) path to folder to save label nrrd
    Returns:
        Either a sitk image object or a numpy array derived from it (depending on 'return_type') of both image and label.
    Raises:
        Exception if an error occurs.
    """
    try:
        # get image, arr, and spacing
        image_obj, image_arr, image_spacing, image_origin = get_arr_from_nrrd(path_to_image_nrrd, "image")
        
        ## Return top 25 rows of 3D volume, centered in x-y space / start at anterior (y=0)?
        print("image_arr shape: ", image_arr.shape)
        c,y,x = image_arr.shape
        
        ## Get center of mass to center the crop in Y plane
        mask_arr = np.copy(image_arr) 
        mask_arr[mask_arr > -500] = 1
        mask_arr[mask_arr <= -500] = 0
        mask_arr[mask_arr >= -500] = 1 
        print("mask_arr min and max:", np.amin(mask_arr), np.amax(mask_arr))
        centermass = ndimage.measurements.center_of_mass(mask_arr) # z,x,y   
        cpoint=c-crop_shape[2]//2
        print("cpoint, ", cpoint)
        centermass = ndimage.measurements.center_of_mass(mask_arr[cpoint,:,:])   
        print("center of mass: ", centermass)
        startx = int(centermass[0] - crop_shape[0]//2)
        starty = int(centermass[1] - crop_shape[1]//2)      
        #startx = x//2 - crop_shape[0]//2       
        #starty = y//2 - crop_shape[1]//2
        startz = int(c - crop_shape[2])
        print("start X,Y,Z: ", startx, starty, startz)
        if startz < 0:
            image_arr = np.pad(image_arr,
                ((abs(startz)//2, abs(startz)//2),
                (0, 0),
                (0, 0)), 'constant', constant_values=-1024)
            image_arr_crop = image_arr[0:crop_shape[2],starty:starty+crop_shape[1],startx:startx+crop_shape[0]]
        else:
            image_arr_crop = image_arr[0:crop_shape[2],starty:starty+crop_shape[1],startx:startx+crop_shape[0]]
        if image_arr_crop.shape[0] < crop_shape[2]:
            print("initial cropped image shape too small:", image_arr_crop.shape)
            print(crop_shape[2],image_arr_crop.shape[0])
            image_arr_crop = np.pad(image_arr_crop,((int(crop_shape[2]-image_arr_crop.shape[0]),0),(0,0),(0,0)),'constant',constant_values=-1024)
            print("padded size: ", image_arr_crop.shape)
        print('Returning bottom rows')
        output_path_image = os.path.join(output_folder_image, "{}_{}_image_interpolated_roi_raw_gt.nrrd".format(dataset, patient_id))
        # save nrrd
        image_crop_sitk = generate_sitk_obj_from_npy_array(image_obj, image_arr_crop, resize=False, output_dir=output_path_image)
        print("Saving image cropped")
        if return_type == "sitk_object":
            return image_crop_sitk
        elif return_type == "numpy_array":
            return image_arr_crop
    except Exception as e:
        print ("Error in {}_{}, {}".format(dataset, patient_id, e))
