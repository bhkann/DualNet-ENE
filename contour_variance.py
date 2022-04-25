### Custom Script to auto-generate 3D random pertubations in lymph node region of interest segmentation ###
### Written by Benjamin H. Kann and Jirapat Likitlersuang, 1/12/2022 ###

def _contour_variance(arr):
    """
    Input Mask
    Returns image with mask variance, mimicking human variation
    """
    mask3d_NEW = np.array(arr)
    ### Set random parameters for GLOBAL EROSION and DILATION ###
    rand_e = randint(1,10)
    rand_d = randint(1,10)

    ## FIND BOUNDING BOX ##
    rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(mask3d_NEW) #was mask3d_err
    for r in range(rmin, rmax+1):
        rNow = mask3d_NEW[r]
        rNow = np.float32(rNow)
        _, threshold = cv2.threshold(rNow, 0, 1, cv2.THRESH_BINARY)
        threshold = threshold.astype(np.uint8)
        contours, _= cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
        listContour = list(contours[0])    
        #print("NEW LIST:" , listContour)
        for c in listContour:  # contours[0]:
            #print(c)
            rand_01 = randint(0,5)
            R_ = r
            G_ = c[0][1]
            Y_ = c[0][0]
            #print([R_, G_, Y_])
            mask3d_NEW[R_, G_-rand_01:G_+rand_01, Y_-rand_01:Y_+rand_01] = 1
    ## Serial erosion, dilation, closing to create smoothing effect ##
    mask3d_NEW = ndimage.binary_erosion(mask3d_NEW, np.ones((1,2+rand_e,2+rand_e)), iterations=1)
    mask3d_NEW = ndimage.binary_dilation(mask3d_NEW, np.ones((1,2+rand_d,2+rand_d)), iterations=1)
    mask3d_NEW = ndimage.binary_closing(mask3d_NEW, np.ones((1,2,2)), iterations=2)
    #arr_new = np.multiply(mask3d_NEW, arr)

    return mask3d_NEW 

def bbox2_3D(img):
    """
    Returns bounding box fit to the boundaries of non-zeros
    """
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