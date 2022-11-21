### Custom Script to auto-generate 3D random pertubations in lymph node region of interest segmentation ###
### Written by Benjamin H. Kann and Jirapat Likitlersuang, 1/12/2022 ###

def _contour_variance(self, arr, select_var):
    """
    Returns image with mask variance, mimicking human variation
    """
    # Load 3d mask
    #sitk_image = sitk.ReadImage(path_to_nrrd)
    #arr = sitk.GetArrayFromImage(sitk_image)
    # main/origianl mask
    #mask3d = np.array(arr)
    # Initialized mask before mophology
    #mask3d_dil = np.array(arr)
    #mask3d_err = np.array(arr)
    mask3d = np.array(arr)
    mask3d[mask3d>0]=1
    # Get coor. boundary of the mask
    #listBoundary = []
    mask3d_NEW = np.array(arr)
    ### GLOBAL EROSION ###
    #random.seed(select_var)
    rand_e = randint(3,6)

    mask3d_err = ndimage.binary_erosion(mask3d, np.ones((1,rand_e,rand_e)), iterations=1)
    #print(np.sum(mask3d_err))

    ## FIND BOUNDING BOX ##
    rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(mask3d_err)
    for r in range(rmin, rmax+1):
        rNow = mask3d_err[r]
        rNow = np.float32(rNow)
        _, threshold = cv2.threshold(rNow, 0, 1, cv2.THRESH_BINARY)
        threshold = threshold.astype(np.uint8)
        contours, _= cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
        listContour = list(contours[0])    
        #print("NEW LIST:" , listContour)
        for c in listContour:  # contours[0]:
            #print(c)
            rand_01 = randint(0,3)
            R_ = r
            G_ = c[0][1]
            Y_ = c[0][0]
            #print([R_, G_, Y_])
            mask3d_NEW[R_, G_-rand_01:G_+rand_01, Y_-rand_01:Y_+rand_01] = 1
    #old way: 
    #mask3d = ndimage.binary_erosion(mask3d, np.ones((1,3,3)), iterations=1)
    #mask3d = ndimage.binary_dilation(mask3d, np.ones((1,2,2)), iterations=1)
    #mask3d = ndimage.binary_closing(mask3d, np.ones((1,2,2)), iterations=2)
    #arr_new = np.multiply(mask3d, arr)
    #new way:
    mask3d_NEW = ndimage.binary_erosion(mask3d_NEW, np.ones((1,2,2)), iterations=1)
    mask3d_NEW = ndimage.binary_dilation(mask3d_NEW, np.ones((1,2,2)), iterations=1)
    mask3d_NEW = ndimage.binary_closing(mask3d_NEW, np.ones((1,2,2)), iterations=2)
    arr_new = np.multiply(mask3d_NEW, arr)

    return arr_new  

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