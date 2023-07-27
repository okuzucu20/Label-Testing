from src.constants import Models, TestTypes
from sys import stderr
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from src.devimg.DeviceImage import DeviceImage
from src.constants import *

PI_NAME = ["power"] # Power indicator name list
CVI_NAME = ["capacitive", "voltage", "indicator"] # capacitive voltage indicator name list
POWER_LED_EXTENSION_Y_PX = 40

class IndicatorImage(DeviceImage):

    def __init__(self, img_filename: str, model: Models, ocr) -> None:
        super().__init__(img_filename, model, test_type=TestTypes.INDICATOR)

        # OCR is taken as an argument in order to eliminate multiple 
        # initialization for different images
        self.ocr = ocr

        self.text_list = None
        self.is_power_on = None
        self.img_display_search = None

    def detect_text(self, img=None):
        # If an image is not given, mask the raw image and assign it as the input image
        if img is None:
            img = self.img_raw #self.mask()

        # Read the text with ocr
        self.text_list = self.ocr.ocr(cv.cvtColor(img, cv.COLOR_BGR2RGB))[0]
        return self.text_list
    
    def field_bbox(self, field_names, extension_y_px):

        if self.text_list is None:
            raise Exception("Texts on the image are not detected before cropping the text area")

        bbox = None
        for text_info in self.text_list:
            text = text_info[1][0]

            if np.sum([(sub_name in text.lower()) for sub_name in field_names]) > 0:
                bbox = [[int(coord) for coord in pt] for pt in text_info[0]]
                break

        if bbox is None:
            print("Cannot find area that is determined by the field names", file=stderr)
            return None
        
        # Change the format of the bbox and extend its height with a constant amount
        bbox = [bbox[0][0],            bbox[0][1], 
                bbox[2][0]-bbox[0][0], bbox[2][1]-bbox[0][1]+extension_y_px]
        return bbox
    
    def power_indicator_bbox(self):
        # Crop the approximate area of led
        power_bbox = self.field_bbox(PI_NAME, POWER_LED_EXTENSION_Y_PX)
        return power_bbox
    
    # Returns the bounding box for searching the display
    def display_search_bbox(self):
        # Find the bbox of the area 
        ds_bbox = self.field_bbox(CVI_NAME, 0)
        if ds_bbox is None:
            return None
        
        ds_bbox[3] = self.img_raw.shape[0] - ds_bbox[1]
        return ds_bbox

    def check_power(self) -> bool:

        bbox = self.power_indicator_bbox()
        if bbox is None:
            self.is_power_on = IndicatorImage.has_bright_point(self.img_raw)
            return self.is_power_on
            #raise Exception("Power indicator not detected before checking power supply")
        
        # Crop the image to only see power indicator part
        power_img = self.img_raw[bbox[1]:bbox[1]+bbox[3],
                                 bbox[0]:bbox[0]+bbox[2]]
        
        self.is_power_on = IndicatorImage.has_bright_point(power_img)
        return self.is_power_on
    
    def display_search_area(self):

        bbox = self.display_search_bbox()
        if bbox is None:
            raise Exception("Field name bbox is not detected before cropping the display area")
        
        self.img_display_search = self.img_raw[bbox[1]:bbox[1]+bbox[3],
                                               bbox[0]:bbox[0]+bbox[2]]
        return self.img_display_search


    @staticmethod
    def has_bright_point(img) -> bool:
        #plt.imshow(cv.cvtColor(cv.inRange(cv.cvtColor(img, cv.COLOR_BGR2HSV)[:,:,2], 250, 255), cv.COLOR_BGR2RGB))
        #plt.show()
        #print(np.sum(cv.inRange(cv.cvtColor(img, cv.COLOR_BGR2HSV)[:,:,2], 250, 255)))
        return np.sum(cv.inRange(cv.cvtColor(img, cv.COLOR_BGR2HSV)[:,:,2], 250, 255)) > 1000



    def detect_corners():
        pass
        


