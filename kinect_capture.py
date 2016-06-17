import cv2
import scipy.stats
from primesense import openni2
import numpy as np

def open_kinect():
    """Open a connection to the first Kinect found, and
    then start capturing depth and color images. Returns handles
    to the depth and color streams."""

    # initialise openNI
    openni_path = r"C:\Program Files (x86)\OpenNI2\Redist"
    #openni_path = r"/Library/OpenNI-MacOSX-x64-2.2/Redist"
    openni2.initialize(openni_path)     # can also accept the path of the OpenNI redistribution

    #devs = openni2.Device.enumerateDevices()
    #print devs
    # open first Kinect
    dev = openni2.Device.open_any()    

    # connect to the depth and color cameras
    depth_stream = dev.create_depth_stream()    
    
    # start the stream capture
    depth_stream.start()
    return depth_stream

def capture_images(depth_stream, size):
    """Capture images from the color and depth streams, and
    return them as arrays."""
    
    w,h = size    
    frame = depth_stream.read_frame()    
    frame_data = frame.get_buffer_as_uint16()    
    depth_image = np.frombuffer(frame_data, dtype=np.uint16, count=w*h)    
    depth_image = depth_image.reshape((h,w))     
    frame._close()
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