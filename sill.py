import numpy as np
import os
import cv2
import time
import config
import scipy.ndimage
import pygame, random
import scipy.stats
from pygame.locals import *
import pygame.gfxdraw
import skeleton
from blob_tracker import BlobTracker
import freenect
from kinect_capture import capture_images, threshold_depth_image
from contours import *

mscale = 4
# Constants for traversing the contour graph
NEXT, PREV, CHILD, PARENT = 0,1,2,3
colors = [[np.random.randint(0,255) for i in range(3)] for j in range(500)]


def make_smoothing_element(close_size):
    """Return a circular smoothing element of the given diameter."""
    return cv2.getStructuringElement(cv2.MORPH_RECT,(close_size, close_size))


def blit_alpha(target, source, location, opacity):
        x = location[0]
        y = location[1]
        temp = pygame.Surface((source.get_width(), source.get_height())).convert()
        temp.blit(target, (-x, -y))
        temp.blit(source, (0, 0))
        temp.set_alpha(opacity)
        target.blit(temp, location)


class Topologic(object):


    def __init__(self):
        self.skeleton = skeleton.Skeleton(tick_fn=self.tick, draw_fn = self.draw, size=(800,600))

        self.tracker = BlobTracker()
        # grab a frame here just to get the resolution
        tmp_frame, _ = freenect.sync_get_depth(format=freenect.DEPTH_MM)
        self.w, self.h = tmp_frame.shape

        # create smoothing kernel
        self.kernel = make_smoothing_element(config.closing_size)

        self.xm, self.ym = np.meshgrid(np.arange(self.w/mscale), np.arange(self.h/mscale))
        self.median_finder = np.zeros((self.h/mscale,self.w/mscale), dtype=np.uint8)
        self.ellipse_finder = np.zeros((self.h/mscale,self.w/mscale), dtype=np.uint8)
        self.draw_surface = pygame.surface.Surface((self.w, self.h))

    def start(self):
        self.skeleton.main_loop()

    def get_contours(self):
        """Get the contours of the image, after thresholding/filtering"""
        depth_image = capture_images((self.w,self.h))

        thresh = threshold_depth_image(depth_image, min_depth=config.min_depth, max_depth=config.max_depth, threshold=config.min_depth, kernel=self.kernel)

        # extract contours
        im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy!=None:
            hierarchy = hierarchy[0]
            return contours, hierarchy
        else:
            return [], []

    def draw(self, screen):
        w,h = screen.get_width(), screen.get_height()
        # blit the main buffer
        if config.rotate_image_deg != 0:
            drawimg = pygame.transform.rotate(self.draw_surface, config.rotate_image_deg)
        else:
            drawimg = self.draw_surface
        screen.blit(drawimg, ((w-self.w)/2, (h-self.h)/2))

    def get_bounding_boxes(self, contours, hierarchy):
        # get bounding boxes
        bboxes = []

        for i,contour in enumerate(contours):
            # if large and top level (i.e. a whole person contour)
            if hierarchy[i][PARENT]==-1 and cv2.contourArea(contour)>config.outer_contour_min_area:
                contour_rect = list(cv2.boundingRect(contour))
                bboxes.append(contour_rect)
        return bboxes

    def update_surfaces(self, simple):
        self.median_finder[:] = 0
        self.ellipse_finder[:] = 0
        cv2.drawContours(self.median_finder, [simple/mscale], -1, 1, -1)

    def draw_contour(self, simple, color):
        pts = [p[0] for p in simple]
        pygame.draw.polygon(self.draw_surface, color,  pts )
        pygame.draw.aalines(self.draw_surface, (0,0,0), True, pts )


    def get_head(self, x, y, x1, x2):
        hx2 = x+(x-x1)
        hx1 = x+(x-x2)
        x_l = int(hx1/mscale)
        x_r = int(hx2/mscale)
        # find top values inside the central column
        fy = (self.xm+1) * np.where(self.median_finder>0,1,1e10)
        top_ys = np.argmin(fy[:,x_l:x_r], axis=0) * mscale
        top_xs = np.arange(x_l, x_r) * mscale

        head_y = np.min(top_ys)
        head_x = (np.argmin(top_ys)*1.1+x_l) * mscale
        # compute linear fit to the head top
        if len(top_xs)==len(top_ys):
            p = np.polyfit(top_xs, top_ys, 1)
        else:
            p = [1,1]
        return head_x, head_y, hx1-hx2, p[0]

    def get_exterior(self, x, y, x1, x2, bottom, head_y):
         """Find the exterior contours, outside the personal bubble"""
         fx1 = x+(x-x1)*8
         fx2 = x+(x-x2)*8
         # compute bounding ellipse; and intersection with body outline
         cv2.ellipse(self.ellipse_finder, ((x/mscale,y/mscale), ((fx1-fx2)/mscale, (2*(bottom-head_y))/mscale), 0), 255,-1 )
         intersection = np.bitwise_and(255-self.ellipse_finder, self.median_finder)
         # find external blobs
         im2, out_contours, out_hierarchy = cv2.findContours(intersection,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
         return out_contours, out_hierarchy, fx1-fx2

    def draw_exterior_contours(self, outlines):
        for out in outlines:
            pts = [p[0]*mscale for p in out]
            pygame.draw.polygon(self.draw_surface, (0,255,0),  pts )
            pygame.draw.aalines(self.draw_surface, (0,0,0), True, pts )

    def draw_bubble(self,x,y,w,h,state):
        out_fade = np.sin(state*np.pi)
        if state<0.5:
            a = [255,255,255,0]
            b = [200,190,130,100]

        else:
            a = [200,190,130,20]
            b = [200,190,130,100]
        c = ((1-out_fade)*np.array(a) + (out_fade)*np.array(b)).astype(np.uint8)
        thick = (1-out_fade)*1 + (out_fade)*4
        #pygame.gfxdraw.aaellipse(self.draw_surface, c, (x-w/2, y-h/2, w,h), int(thick))
        pygame.gfxdraw.aaellipse(self.draw_surface, int(x), int(y), int(w/2),int(h/2), c)
        pygame.gfxdraw.filled_ellipse(self.draw_surface, int(x), int(y), int(w/2),int(h/2), c)


    def tick(self, dt):
        # clear the drawing buffer
        self.draw_surface.fill((255,255,255))
        # update the animations for later drawing
        # for anim in self.live_animations:
        #     anim.tick(dt)

        contours, hierarchy = self.get_contours()
        bboxes = self.get_bounding_boxes(contours, hierarchy)
        # update uids for each contour
        uids = self.tracker.match(bboxes)

        tracked_contours = []
        j = 0
        for i,contour in enumerate(contours):
            # if large and top level (i.e. a whole person contour)
            if hierarchy[i][PARENT]==-1 and cv2.contourArea(contour)>config.outer_contour_min_area:

                uid = uids[j]
                # get the approximated contours
                simple = cv2.approxPolyDP(contour, 2, True)
                self.update_surfaces(simple)

                # compute bubble ellipse
                x, y = get_percentiles(self.median_finder, self.xm, self.ym, 50, mscale=mscale)
                x1, y1 = get_percentiles(self.median_finder, self.xm, self.ym, 35, mscale=mscale)
                x2, y2 = get_percentiles(self.median_finder, self.xm, self.ym, 65, mscale=mscale)

                bottom = np.max([c[0][1] for c in simple])

                # get head co-ordinates
                head_x, head_y, head_width, head_gradient = self.get_head(x,y,x1,x2)

                # approximate arm level (i.e. upper/lower division line)
                arm_level = y-(y-head_y)*0.2

                # get exterior contours
                out_contours, out_hierarchy, bubble_width = self.get_exterior(x,y,x1,x2,bottom,head_y)
                out_quadrants, outlines = [], []
                if len(out_contours)>0:
                    out_hierarchy = out_hierarchy[0]
                    out_quadrants,outlines = get_quadrants(out_contours, out_hierarchy, 0, (x/mscale, arm_level/mscale), config.external_contour_min_area/(mscale))

                # get hole gesture quadrants
                quadrants, holes = get_quadrants(contours, hierarchy, hierarchy[i][CHILD], (x,arm_level), config.hole_contour_min_area)

                if config.draw_bubbles:
                    self.draw_bubble(x,y, bubble_width, 2*(bottom-head_y), 0.5)
                # draw body silhouette, and the hat, if there is one
                self.draw_contour(simple, colors[uid % len(colors)])

                j = j + 1


    def quit(self):
        pass


t = Topologic()
t.start()

