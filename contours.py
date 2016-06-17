import cv2
import numpy as np

NEXT, PREV, CHILD, PARENT = 0,1,2,3

def get_strip(hierarchy,i):
    """Return every element in this sequence of nodes, following
    the next pointers, and recursing into children to find sub-nodes."""
    strip = []
    while i!=-1:
        # any children?
        if hierarchy[i][CHILD]==-1:
            children = []
        else:
            # if so, create a strip for them
            children = get_strip(hierarchy, hierarchy[i][CHILD])            
        strip.append((i, children))
        # move to next contour in the hierarchy
        i = hierarchy[i][NEXT]
    return strip
    
    
    
def contour_hierarchy(hierarchy):
    """Take a contour, hierarchy pair, and return a tree as a nested list
    with all of the elements inside"""
    return get_strip(hierarchy, 0)
                     
    
def get_gesture_type(quadrants):
    # recognize hole gestures
    gesture,symmetric,up = False,False,False
    if len(quadrants)==1:
        symmetric = False
        if quadrants[0]<2:
            up = True
        gesture = True
        
    if len(quadrants)==2:
        symmetric = True
        if quadrants[0]<2 and quadrants[1]<2:
            up = True
            gesture = True
        if quadrants[0]>1 and quadrants[1]>1:
            up = False
            gesture = True
    return gesture, symmetric, up

def get_top_intersection(contour, x_line):
    min_y = None
    min_x = None
    last_x = None
    for c in contour:
        x,y = c[0][0], c[0][1]        
        # find intersection
        if last_x!=None and (x>=x_line and last_x<=x_line) or (x<=x_line and last_x>=x_line):
            if min_y==None or y<min_y:
                min_y = y
                min_x = x
        last_x = x
    return min_x, min_y
    
    

def get_centroid(contour):
    """Return the centroid of the given contour"""
    moments = cv2.moments(contour)
    cx = moments['m10'] / moments['m00'] 
    cy = moments['m01'] / moments['m00'] 
    return cx,cy
    

def get_percentiles(mask, xm, ym, percentile, mscale):
        x1 =  np.percentile(xm[mask>0],percentile)
        y1 =  np.percentile(ym[mask>0],percentile)
        return x1*mscale,y1*mscale

def quadrant_code(center, pos):
    x,y = center
    cx,cy = pos
    # compute angle of line between outline center and hole center
    # and code the quadrant as q in the pattern
    # 0 | 1
    # 2 | 3
    angle = np.arctan2((cx-x), (cy-y)) / np.pi * 180.0
    q = 0
    if angle<0:
        q = 1
        angle = -angle
    angle = angle - 90
    if angle<0:
        q = q + 2                    
    return q
        
def get_quadrants(contours, hierarchy, i, center, min_area):    
    if i==-1:
        return [], []        
    quadrants = []
    x,y = center
    holes = []
    # look for holes        
    next = i    
    while next!=-1:                                   
        # if while is reasonably big
        if cv2.contourArea(contours[next])>min_area:
            cx, cy = get_centroid(contours[next])                                                
            holes.append(contours[next])
            q = quadrant_code(center, (cx,cy))
            quadrants.append(q)                                              
            # move to next hole inside this outer contour
        next = hierarchy[next][NEXT]                                    
    return quadrants, holes    