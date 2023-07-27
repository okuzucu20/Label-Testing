from abc import ABC, abstractmethod
from src.constants import *
import cv2 as cv
import os
import numpy as np

class DeviceImage(ABC):

    def __init__(self, img_filename: str, model: Models, test_type: TestTypes) -> None:
        super().__init__()

        # Original image, read initially
        self.img_name = img_filename
        self.model = model
        self.test_type = test_type
        self.img_raw = cv.imread(self.file_raw_path)

        # Transformed images
        self.img_gray_blur = None
        self.img_masked = None
        self.img_contour = None
        self.img_perspective = None

        # Last calculated (transformed) image type that can be saved (that has its own subfolder)
        self.last_calc_img = None

        # Intermediate values for getting transformed images
        self.contours = None
        self.contour_largest = None
        self.corner_pts = None
        self.sketch_img = None
        self.reset_sketch()

    #### Image Processing Functions ####
    def pre_process(self, img=None):

        # Select the image to transform
        img = self.img_raw if img is None else img

        self.img_gray_blur = cv.GaussianBlur(cv.cvtColor(img, cv.COLOR_BGR2GRAY), (3, 3), 0)
        return self.img_gray_blur
    
    def mask(self, img=None):

        # Select the image to transform
        if img is None:
            if self.img_gray_blur is None:
                img = self.img_raw
            else:
                img = self.img_gray_blur

        if len(img.shape) > 2:
            img = self.pre_process(img)

        self.img_masked = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,51,7)
        self.last_calc_img = ImageTypes.THRESH
        return self.img_masked
    
    def detect_edge(self, img=None):

        # Select the image to transform
        if img is None:
            if self.img_perspective is None:
                img = self.img_raw
            else:
                img = cv.cvtColor(self.img_perspective, cv.COLOR_BGR2GRAY)

        self.img_canny = cv.Canny(img, 200, 250, apertureSize=3)
        self.last_calc_img = ImageTypes.EDGE
        return self.img_canny
    
    def detect_contours(self, sketch_img=None, return_all=False):

        if self.img_masked is None:
            raise Exception("Masked image is not calculated before contour detection")
        img = self.img_masked
        
        if sketch_img is None:
            sketch_img = self.sketch_img

        self.contours, hierarchy = cv.findContours(image=img, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
        self.contour_largest = sorted(self.contours, key=cv.contourArea, reverse=True)[0]

        # Draw the contours for the edges
        cv.drawContours(image=sketch_img, contours=self.contours if return_all else [self.contour_largest], 
                        contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)

        self.img_contour = sketch_img
        self.sketch_img = sketch_img
        self.last_calc_img = ImageTypes.CONTOUR

        return sketch_img, self.contours if return_all else self.contour_largest
    
    @abstractmethod
    def detect_corners(self, sketch_img):
        pass

    def perspective_transform(self, img=None, out_dims=(800,400)):

        # Select the image to transform
        img = self.img_raw if img is None else img

        if self.corner_pts is None:
            raise Exception("Corner points for the label is not calculated before perspective transform")
        corner_pts = self.corner_pts

        # Get the birdseye view perspective transform
        input_pts = np.float32(corner_pts.reshape((4,2)))
        output_pts = np.float32([[0, 0],
                                [out_dims[0]-1, 0],
                                [out_dims[0]-1, out_dims[1]-1],
                                [0, out_dims[1]-1]])
        pM = cv.getPerspectiveTransform(input_pts, output_pts)

        # Apply the transform
        self.img_perspective = cv.warpPerspective(img, pM, out_dims, flags=cv.INTER_LINEAR)
        self.last_calc_img = ImageTypes.PERSPECTIVE

        return self.img_perspective
    
    @property
    def test_path(self):
        return os.path.join(IMG_FOLDER, self.model.value, self.test_type.value)

    @property
    def file_raw_path(self):
        return os.path.join(self.test_path, ImageTypes.RAW.value, self.img_name)
    
    @property
    def file_thresh_path(self):
        return os.path.join(self.test_path, ImageTypes.THRESH.value, self.img_name)
    
    @property
    def file_contour_path(self):
        return os.path.join(self.test_path, ImageTypes.CONTOUR.value, self.img_name)
    
    @property
    def file_edge_path(self):
        return os.path.join(self.test_path, ImageTypes.EDGE.value, self.img_name)

    @property
    def file_perspective_path(self):
        return os.path.join(self.test_path, ImageTypes.PERSPECTIVE.value, self.img_name)
    
    @property
    def file_segment_path(self):
        return os.path.join(self.test_path, ImageTypes.SEGMENT.value, self.img_name)
    
    def save(self):
        # Save the last calculated image
        lci = self.last_calc_img
        if lci == ImageTypes.THRESH:
            cv.imwrite(self.file_thresh_path, self.img_masked)
        elif lci == ImageTypes.CONTOUR:
            cv.imwrite(self.file_contour_path, self.img_contour)
        elif lci == ImageTypes.EDGE:
            cv.imwrite(self.file_edge_path, self.img_canny)
        elif lci == ImageTypes.PERSPECTIVE:
            cv.imwrite(self.file_perspective_path, self.img_perspective)
        elif lci == ImageTypes.SEGMENT:
            cv.imwrite(self.file_segment_path, self.img_segment)

    def reset_sketch(self):
        self.sketch_img = self.img_raw.copy()
