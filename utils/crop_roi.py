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