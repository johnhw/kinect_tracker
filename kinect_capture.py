import cv2
import scipy.stats
import numpy as np
import freenect

def capture_images(size):
    """Capture images from the color and depth streams, and
    return them as arrays."""
    w,h = size    
    frame_data, ts = freenect.sync_get_depth(format=freenect.DEPTH_MM)
    # note: image still seems to be rotated 90 degrees after this
    depth_image = frame_data.reshape((w,h)).T 
    return depth_image    
    
kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))    

def threshold_depth_image(depth_image, min_depth, max_depth, threshold, kernel=None):
    """ Return a thresholded and cleaned version of the depth image, as uint8 array.
    Takes a depth image, and zeros any values <= min_depth and >= max_depth.
    Thresholds this image at z=threshold (z>threshold=1, z<threshold=0).
    Applies the closing operation with the given kernel to the resulting image to clean up small defects."""
    # clear away too near/too far values
    
    depth_image = scipy.stats.threshold(depth_image, min_depth, max_depth, 0)            
    # binary threshold (all values >70cm)       
    if kernel!=None:
        depth_image = cv2.morphologyEx(depth_image, cv2.MORPH_CLOSE, kernel)
    uint_depth = (depth_image/10.0).astype(np.uint8)
    thresh = cv2.Canny(uint_depth, 100, 200, 32)                               
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel_small)
    rv,thresh2 = cv2.threshold(uint_depth, threshold/10, 1, cv2.THRESH_BINARY)        
    cv2.GaussianBlur(thresh2, (0,0),  5, thresh2)
    
    thresh2 = ((~thresh) & thresh2)        
    cv2.GaussianBlur(thresh2, (0,0),  1, thresh2)
    
    return thresh2  
    
    
def threshold_depth_image_smooth(depth_image, min_depth, max_depth, threshold, kernel=None):
    """ Return a thresholded and cleaned version of the depth image, as uint8 array.
    Takes a depth image, and zeros any values <= min_depth and >= max_depth.
    Thresholds this image at z=threshold (z>threshold=1, z<threshold=0).
    Applies the closing operation with the given kernel to the resulting image to clean up small defects."""
    # clear away too near/too far values
    
    depth_image = scipy.stats.threshold(depth_image, min_depth, max_depth, 0)            
    # binary threshold (all values >70cm)       
    if kernel!=None:
        depth_image = cv2.morphologyEx(depth_image, cv2.MORPH_CLOSE, kernel)
    uint_depth = (depth_image/10.0).astype(np.uint8)
    thresh = cv2.Canny(uint_depth, 100, 200, 32)                               
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel_small)
    rv,thresh2 = cv2.threshold(uint_depth, threshold/10, 1, cv2.THRESH_BINARY)        
    cv2.GaussianBlur(thresh2, (0,0),  5, thresh2)
    
    thresh2 = ((~thresh) & thresh2)        
    cv2.GaussianBlur(thresh2, (0,0),  3, thresh2)
    
    return thresh2  
