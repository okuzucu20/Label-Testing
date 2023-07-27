#!/usr/bin/env python
from enum import Enum
from math import atan2
import cv2 as cv
import os
import numpy as np
import easyocr as ocr
import qr2text
from paddleocr import PaddleOCR
#import keras_ocr as ocr
import matplotlib.pyplot as plt

IMG_FOLDER = "PicturesOld"

class ImageTypes(Enum):
    RAW = "raw"
    THRESH = "threshold"
    CONTOUR = "contour"
    EDGE = "edge"
    PERSPECTIVE = "perspective"
    SEGMENT = "segment"

class ImageTypePaths(Enum):
    RAW = os.path.join(IMG_FOLDER, ImageTypes.RAW.value)
    THRESH = os.path.join(IMG_FOLDER, ImageTypes.THRESH.value)
    CONTOUR = os.path.join(IMG_FOLDER, ImageTypes.CONTOUR.value)
    EDGE = os.path.join(IMG_FOLDER, ImageTypes.EDGE.value)
    PERSPECTIVE = os.path.join(IMG_FOLDER, ImageTypes.PERSPECTIVE.value)
    SEGMENT = os.path.join(IMG_FOLDER, ImageTypes.SEGMENT.value)

class Models(Enum):
    THC100D = 1
    OVI = 2

class TestTypes(Enum):
    LABEL = 1
    INDICATOR = 2

QR_SEGMENT_INDICES = {Models.THC100D: 4}
CORRECTION_INDICES = {Models.THC100D: {0: None, 5: "key"}} # keys indicate segment indices
                                                           # if value is None, change full text
                                                           # elif value is "key" change the part before ":" in the text (if found)
                                                           # elif value is "val" change the part after  ":" in the text (if found)
#GROUND_TRUTH = {Models.THC100D: []}

#### Helper Functions ####
def intersection(L1_pts, L2_pts):

    def line(p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C
    
    L1 = line(L1_pts[0], L1_pts[1])
    L2 = line(L2_pts[0], L2_pts[1])

    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return np.array([x,y])
    else:
        return False

'''
Calculate the corners for the quad that is enclosing the label contour
'''
def get_corner_pixels(closed_contour, use_4_direction=False):

    # Calculate the 8 cardinal points to find an octogonal shape that is approximately enclosing the label
    cardinal_points = []
    for i in range(-1,2,1):
        for j in range(-1,2,1):
            if i == 0 and j == 0:
                continue
            
            multiplier = [[i],[j]]
            max_point = [[0,0]]
            max_value = float('-inf')

            for p in closed_contour:
                value = p @ multiplier
                if value > max_value:
                    max_point = p
                    max_value = value
            
            cardinal_points.append(max_point)

    #[-1,-1], [-1,0], [-1,1], [0,-1], [0,1], [1,-1], [1,0], [1,1]
    # Change the indices to get a convex shape
    octogonal_indices = [6, 7, 4, 2, 1, 0, 3, 5]
    cardinal_points = np.array(cardinal_points)[octogonal_indices]

    # Calculate the lengths of each edge (octogonal shape)
    side_lengths = np.sqrt(np.sum(np.square(np.roll(cardinal_points, -1, axis=0) - cardinal_points), axis=2))

    # Find the largest 4 edges and the points that determine them
    longest_sides = np.sort(np.argpartition(side_lengths.T[0], -4)[-4:])
    side_point_indices = np.array([longest_sides, (longest_sides+1)%8]).T
    side_points = cardinal_points[side_point_indices].reshape(4,2,2)

    # Get the enclosing quad by taking the intersections of the lines on octogonal shape
    # Optional corner points gets only the 4 points (-x-y, -x+y, x-y, x+y).
    corner_points = []; corner_points_opt = []
    for i in range(4):
        corner_points.append([intersection(side_points[i]/100, side_points[(i+1)%4]/100)*100])
        corner_points_opt.append(cardinal_points[2*i+1])

    # Change the indices of elements such that the first-to-second element edge is the largest edge of the quad 
    # (so that later we can apply perspective transform)
    quad_side_lengths = np.sqrt(np.sum(np.square(np.roll(corner_points, -1, axis=0) - corner_points), axis=2)).reshape(4)
    corner_points = np.roll(corner_points, -np.argmax(quad_side_lengths), axis=0)

    quad_side_lengths_opt = np.sqrt(np.sum(np.square(np.roll(corner_points_opt, -1, axis=0) - corner_points_opt), axis=2)).reshape(4)
    corner_points_opt = np.roll(corner_points_opt, -np.argmax(quad_side_lengths_opt), axis=0)

    return (np.array(corner_points, dtype=int) if not use_4_direction else np.array(corner_points_opt, dtype=int))

class LabelImage:

    def __init__(self, img_filename: str, model: Models, 
                 qr_detector=cv.QRCodeDetector(), ocr=PaddleOCR(use_angle_cls=True, lang='en')) -> None:

        # Original image, read initially
        self.img_name = img_filename
        self.img_raw = cv.imread(self.file_raw_path)
        self.model = model

        # QR Detector and OCR are given as arguments in order to eliminate multiple 
        # initialization of these classes for different images
        self.qr_detector = qr_detector
        self.ocr = ocr

        # Transformed images
        self.img_gray_blur = None
        self.img_masked = None
        self.img_contour = None
        self.img_perspective = None
        self.img_canny = None
        self.img_segment = None

        # Last calculated (transformed) image type that can be saved (that has its own subfolder)
        self.last_calc_img = None

        # Intermediate values for getting transformed images
        self.contours = None
        self.contour_largest = None
        self.corner_pts = None
        self.sketch_img = None
        self.reset_sketch()

        # Test information
        self.qrcode_info = None # [decodedText, points, outImage]
        self.segments = None # Holds the bbox for each segment of the label
        self.segment_text_list = None # Each index has a list of texts included inside a segment, 
                                      # that are aligned horizontaly from left to right
        self.segment_text_merged = None # Merged text that is included inside a segment
        self.segment_text_scores = None # Scores for the detected texts

    '''
    Sort segments such that if the top line of a segment is at an upper level than other segments, it has an higher value,
    unless their lines approximately match their height (checked via threshold). In which case, the segment on the left
    has an heigher value.
    '''
    def sort_segments(self):

        if self.segments is None:
            raise Exception("Segments should be calculated before sorting them")
        
        level_threshold_px = 20

        # Assign values for means of sorting
        segment_values = []
        for s in self.segments:
            val = [s[0] + s[2]/2, s[1]]
            segment_values.append(val)
        
        unsorted_indices = list(range(len(self.segments)))
        sorted_segments = []
        for _ in range(len(self.segments)):

            # Initialize means of sorting 
            min_mid_x_px = float('inf')
            min_top_y_px = float('inf')
            min_idx = -1

            # Only sort the remaining indices
            for j in unsorted_indices:

                # Assign current value and check whether it has an higher value
                val = segment_values[j]
                if val[1] < min_top_y_px - level_threshold_px or\
                   (abs(val[1] - min_top_y_px) <= level_threshold_px and\
                                        val[0] <  min_mid_x_px):
                    min_idx = j
                    min_mid_x_px = val[0]
                    min_top_y_px = val[1]

            sorted_segments.append(self.segments[min_idx])
            unsorted_indices.remove(min_idx)


        self.segments = sorted_segments
        return self.segments

    #### Image Processing Functions ####
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

    def detect_corners(self, sketch_img=None, use_4_direction=False):

        if self.contour_largest is None:
            raise Exception("Contour for the label is not calculated before corner detection")
        contour = self.contour_largest

        if sketch_img is None:
            sketch_img = self.sketch_img

        # Find the corners and draw the corresponding circles
        self.corner_pts = get_corner_pixels(contour, use_4_direction)
        for p in self.corner_pts:
            sketch_img = cv.circle(sketch_img, p[0], radius=20, color=(255*use_4_direction, 0, 255), thickness=-1)

        self.img_contour = sketch_img
        self.sketch_img = sketch_img
        self.last_calc_img = ImageTypes.CONTOUR

        return sketch_img, self.corner_pts

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

    def mask_image(self, img=None):

        # Select the image to transform
        if img is None:
            if self.img_gray_blur is None:
                img = self.img_raw
            else:
                img = self.img_gray_blur

        if len(img.shape) > 2:
            img = self.pre_process_image(img)

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
    
    def correct_orientation(self):

        if self.qrcode_info[1] is None:
            raise Exception("QR Code should be decoded before orientation correction")

        if self.img_perspective is None:
            raise Exception("Calculate the perspective image or provide another one before correcting the orientation")
        img = self.img_perspective

        if np.mean(self.qrcode_info[1][0], axis=0)[0] < img.shape[1]/2:
            self.img_perspective = cv.rotate(img, cv.ROTATE_180)

        self.last_calc_img = ImageTypes.PERSPECTIVE
        return self.img_perspective

    def pre_process_image(self, img=None):

        # Select the image to transform
        img = self.img_raw if img is None else img

        self.img_gray_blur = cv.GaussianBlur(cv.cvtColor(img, cv.COLOR_BGR2GRAY), (3, 3), 0)
        return self.img_gray_blur
    
    def detect_qrcode(self):

        if self.img_perspective is None:
            raise Exception("Perspective transformed image is not calculated before qrcode detection")
        img = self.img_perspective

        self.qrcode_info = self.qr_detector.detectAndDecode(img)
        return self.qrcode_info
    
    def divide(self, img=None):

        if img is None:
            if self.img_perspective is None:
                raise Exception("Perspective transformed image is not calculated before dividing the label into segments")
            img = self.img_perspective

        # Pre-process and mask the given image
        self.mask_image(img=self.pre_process_image(img=img))

        # Expand the image in all directions with constant value of 255(white)
        border = (5, 10) # pixel width for constant borders (top/bottom, left/right)
        img_border = cv.copyMakeBorder(self.img_masked, border[0], border[0], border[1], border[1], cv.BORDER_CONSTANT, None, value = 255)

        # Used for detecting contours after line pixels are drawn
        binary_img = np.zeros(img_border.shape, np.uint8)

        ### Detect vertical and horizontal lines on masked image ###
        lines_v = cv.HoughLinesP(
            img_border, 1, np.pi, threshold=100, minLineLength=65, maxLineGap=1)
        lines_h = cv.HoughLinesP(
            img_border, 1, np.pi / 2, threshold=500, minLineLength=500, maxLineGap=1)
        
        ### Draw vertical and horizontal lines on both the sketch and binary images ###
        for line in lines_v:
            cv.line(binary_img,line[0][:2],line[0][2:4],255,3)
        for line in lines_h:
            cv.line(binary_img,line[0][:2],line[0][2:4],255,3)

        # Minimum area for a contour to be treated as a field
        field_min_area = 100

        # Find the contours on the image
        contours, hierarchy = cv.findContours(binary_img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

        # First filter according to the min. area, then discard the outer contour
        contours = list(filter(lambda contour: cv.contourArea(contour) > field_min_area, contours))
        contours.remove(max(contours, key=cv.contourArea))

        # Draw the contours onto binary image
        cv.drawContours(image=binary_img, contours=contours, 
                        contourIdx=-1, color=125, thickness=2, lineType=cv.LINE_AA)
        
        # Get the bounding boxes and draw them
        self.segments = []
        for cnt in contours:

            # Get the bbox corresponding to the contour
            bbox = list(cv.boundingRect(cnt))

            # Draw according to border coordinates
            x, y, w, h = bbox
            binary_img = cv.rectangle(binary_img, (x, y), (x + w, y + h), 60, 2)

            # Correct the pixel locations of the bbox according to borderless image
            bbox[0] = max(bbox[0] - border[1], 0)
            bbox[1] = max(bbox[1] - border[0], 0)
            bbox[2] = bbox[2] if bbox[0] + bbox[2] < img.shape[1] else img.shape[1] - bbox[0]
            bbox[3] = bbox[3] if bbox[1] + bbox[3] < img.shape[0] else img.shape[0] - bbox[1]

            # Store bbox
            self.segments.append(bbox)

        # Sort the segments according to their locations
        self.sort_segments()

        text_offset = 5

        # Put the sorted segment numbers on the image
        for i, bbox in enumerate(self.segments):

            # bottom left of the text we want to put
            org = [bbox[0] + int(bbox[2]/2) - text_offset,
                   bbox[1] + int(bbox[3]/2) + text_offset]

            binary_img = cv.putText(binary_img, str(i+1), org, cv.FONT_HERSHEY_SIMPLEX, 
                                    0.5, 255, 2, cv.LINE_AA)


        self.img_segment = binary_img
        self.last_calc_img = ImageTypes.SEGMENT

        return self.img_segment
    
    def correct_text_full(self, img):

        # Find the texts in the image at once (not for each segment)
        text_list = self.ocr.ocr(cv.cvtColor(img, cv.COLOR_BGR2RGB))[0]

        # Checks whether a rectangle ([x,y,w,h]) bounding box contains a point([x,y])
        def rectContains(rect, pt):
            return rect[0] < pt[0] < rect[0]+rect[2] and rect[1] < pt[1] < rect[1]+rect[3]
        
        # Each segment contains a list of texts, initialize a null list
        segment_text_infos = [None] * len(self.segments)

        # Container for the new merged correction texts
        segment_text_merged_new = [None] * len(self.segments)

        for text_info in text_list:

            bb_pts_text, text, score = text_info[0], text_info[1][0], text_info[1][1]

            # 0 and 2 indexed elements are upper left and lower right points respectively
            mid_pt = ((bb_pts_text[0][0] + bb_pts_text[2][0]) / 2,
                      (bb_pts_text[0][1] + bb_pts_text[2][1]) / 2)

            # We haven't found which segment the detected text is in, at the beginning
            found_segment = False
            for i, bb_segment in enumerate(self.segments):

                # If the segment bounding box contains the mid point of text box then we have found the segment
                if rectContains(bb_segment, mid_pt):
                    if segment_text_infos[i] is None:
                        segment_text_infos[i] = [(text, mid_pt)]
                    else:
                        segment_text_infos[i].append((text, mid_pt))

                    found_segment = True
                    break
                
            if not found_segment:
                raise Exception("Could not found segment for the text obtained by full image recognition")
            
        # Sort the texts inside the segments such that their respective bounding boxes 
        # are aligned left to right, then merge the texts
        for i, sti in enumerate(segment_text_infos):
            if sti is None:
                continue

            # Sort the texts
            sti = [sub_ti[0] for sub_ti in sorted(sti, key=(lambda sub_ti: sub_ti[1][0]))]

            # Merge them
            segment_text_merged_new[i] = " ".join(sti)

        # Correct (change) the texts according to correction indices and types
        ci_dict = CORRECTION_INDICES[self.model]
        for ci in ci_dict:
            colon_idx_old = self.segment_text_merged[ci].find(':')
            colon_idx_new = segment_text_merged_new[ci].find(':')

            if ci_dict[ci] is None:
                self.segment_text_merged[ci] = segment_text_merged_new[ci]
            elif ci_dict[ci] == "key":
                if colon_idx_old == -1 or colon_idx_new == -1:
                    print(f"Could not found the key for the segment with index {ci+1}")

                # Change the text before ":"
                self.segment_text_merged[ci] = segment_text_merged_new[ci][:colon_idx_new] +\
                                               self.segment_text_merged[ci][colon_idx_old:]
                                               
            elif ci_dict[ci] == "val":
                if colon_idx_old == -1 or colon_idx_new == -1:
                    print(f"Could not found the value for the segment with index {ci+1}")

                # Change the text after ":"
                self.segment_text_merged[ci] = self.segment_text_merged[ci][:colon_idx_old] +\
                                               segment_text_merged_new[ci][colon_idx_new:]
            else:
                raise Exception("Wrong correction type is specified")
    
    def detect_text(self, img=None, correction="none", preprocess_function=None):

        if img is None:
            if self.img_perspective is None:
                raise Exception("Perspective transformed image is not calculated before text detection")
            img = self.img_perspective

        segments = list(zip(list(range(len(self.segments))), self.segments))
        if correction == "none":
            self.segment_text_list = [None] * len(self.segments)
            self.segment_text_merged = [None] * len(self.segments)
            self.segment_text_scores = [None] * len(self.segments)
        elif correction == "full":
            self.correct_text_full(img)
            return
        elif correction == "segment":
            #segments = [segments[i] for i in CORRECTION_INDICES[self.model].keys()]
            return
        else:
            raise Exception("Wrong correction type is given as an argument")

        def detection_preprocess(img):
            return 255 - self.mask_image(img=self.pre_process_image(img=img))

        for i, bbox in segments:

            if i == QR_SEGMENT_INDICES[self.model]:
                continue

            # Crop image to see only the current segment
            cropped_img = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

            # Pre-process and mask the cropped image
            cropped_img = 255 - self.mask_image(img=cropped_img)

            # Read the text with ocr
            text_list = self.ocr.ocr(cv.cvtColor(cropped_img, cv.COLOR_BGR2RGB))[0] # x_ths=100, y_ths=100, width_ths=100, height_ths=100, ycenter_ths=100
            #print(text_list)

            # Sort the texts in the list according to their individual bbox locations (middle x coordinate)
            text_list_sorted = sorted(text_list, key=(lambda text_info: (text_info[0][0][0] + text_info[0][2][0])/2))

            text_merged = ""; score_mean=0; j=0
            for text_info in text_list_sorted:
                bb, text, score = text_info[0], text_info[1][0], text_info[1][1]
                j+=1
                #print(str(j) + ":", text)

                # Update the mean score for the segment and merge the text
                score_mean += score
                text_merged += text + " "

            #print(text_merged, end="\n\n")
            
            if correction != "none" and score_mean < self.segment_text_scores[i]:
                continue

            # Update the segment values
            self.segment_text_list[i] = text_list_sorted
            self.segment_text_merged[i] = text_merged[:-1]
            self.segment_text_scores[i] = score_mean


            #plt.imshow(cv.cvtColor(cropped_img, cv.COLOR_BGR2RGB))
            #plt.show()

        #print("\n")

    def print_ascii_label(self):
        if self.model == Models.THC100D:
            self.print_ascii_label_thc100d()

    def print_ascii_label_thc100d(self):
        label_length = 80
        qr_one_line_len = 22
        print(self.qrcode_info[0])
        qr_ascii = qr2text.QR(2).from_text(self.qrcode_info[0]).to_ascii_art(trim=True)

        print("-" * label_length)
        print(f"|" + " "*(label_length -2) + "|")
        print(f"|   {self.segment_text_merged[0]}" + " "*(label_length - len(self.segment_text_merged[0]) -5) + "|")
        print(f"|" + " "*(label_length -2) + "|")

        print("-" * label_length)
        print(f"|   " + " "*(label_length//2 -5) + "|" + " "*(label_length//2 -1) + "|")
        print(f"|   {self.segment_text_merged[1]}" + " "*(label_length//2 - len(self.segment_text_merged[1]) -5) + "|", end="")
        print(f"   {self.segment_text_merged[2]}" + " "*(label_length//2 - len(self.segment_text_merged[2]) -4) + "|")
        print(f"|   " + " "*(label_length//2 -5) + "|" + " "*(label_length//2 -1) + "|")

        print("-" * label_length)
        #print(f"|   " + " "*(label_length - qr_one_line_len -9) + "|" + " "*(qr_one_line_len+3) + "|")
        print(f"|   " + " "*(label_length - qr_one_line_len -9) + "|", end="")
        print(f" ", qr_ascii[:qr_one_line_len-1], " |")
        print(f"|   {self.segment_text_merged[3]}" + " "*(label_length - len(self.segment_text_merged[3]) - qr_one_line_len -9) + "|", end="")
        print(f" ", qr_ascii[qr_one_line_len:2*qr_one_line_len-1], " |")
        print(f"|   " + " "*(label_length - qr_one_line_len -9) + "|", end="")
        print(f" ", qr_ascii[2*qr_one_line_len:3*qr_one_line_len-1], " |")

        print("-" * (label_length - qr_one_line_len -5), end="|")
        print(f" ", qr_ascii[3*qr_one_line_len:4*qr_one_line_len-1], " |")

        #print(f"|   " + " "*(label_length - qr_one_line_len -9) + "|" + " "*(qr_one_line_len+3) + "|")
        print(f"|   " + " "*(label_length - qr_one_line_len -9) + "|", end="")
        print(f" ", qr_ascii[4*qr_one_line_len:5*qr_one_line_len-1], " |")
        print(f"|   {self.segment_text_merged[5]}" + " "*(label_length - len(self.segment_text_merged[5]) - qr_one_line_len -9) + "|", end="")
        print(f" ", qr_ascii[5*qr_one_line_len:6*qr_one_line_len-1], " |")
        print(f"|   " + " "*(label_length - qr_one_line_len -9) + "|", end="")
        print(f" ", qr_ascii[6*qr_one_line_len:7*qr_one_line_len-1], " |")

        print("-" * (label_length - qr_one_line_len -5), end="|")
        print(f" ", qr_ascii[7*qr_one_line_len:8*qr_one_line_len-1], " |")

        #print(f"|   " + " "*(label_length - qr_one_line_len -9) + "|" + " "*(qr_one_line_len+3) + "|")
        print(f"|   " + " "*(label_length - qr_one_line_len -9) + "|", end="")
        print(f" ", qr_ascii[8*qr_one_line_len:9*qr_one_line_len-1], " |")
        print(f"|   {self.segment_text_merged[6]}" + " "*(label_length - len(self.segment_text_merged[6]) - qr_one_line_len -9) + "|", end="")
        print(f" ", qr_ascii[9*qr_one_line_len:10*qr_one_line_len-1], " |")
        print(f"|   " + " "*(label_length - qr_one_line_len -9) + "|", end="")
        print(f" ", qr_ascii[10*qr_one_line_len:11*qr_one_line_len-1], " |")

        print("-" * label_length)
        print(f"|" + " "*(label_length -2) + "|")
        print(f"|   " + str(self.segment_text_merged[7]).center(label_length-10) + " " * 5 + "|")
        print(f"|" + " "*(label_length -2) + "|")
        print("-" * label_length)
            

    @property
    def file_raw_path(self):
        return os.path.join(ImageTypePaths.RAW.value, self.img_name)
    
    @property
    def file_thresh_path(self):
        return os.path.join(ImageTypePaths.THRESH.value, self.img_name)
    
    @property
    def file_contour_path(self):
        return os.path.join(ImageTypePaths.CONTOUR.value, self.img_name)
    
    @property
    def file_edge_path(self):
        return os.path.join(ImageTypePaths.EDGE.value, self.img_name)

    @property
    def file_perspective_path(self):
        return os.path.join(ImageTypePaths.PERSPECTIVE.value, self.img_name)
    
    @property
    def file_segment_path(self):
        return os.path.join(ImageTypePaths.SEGMENT.value, self.img_name)
    
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


def main():

    # Initialize qr detector and ocr
    qr_detector = cv.QRCodeDetector()
    reader = PaddleOCR(lang='en') #ocr.Reader(['en', 'tr']) 'SVTR_LCNet'
    model = Models.THC100D

    for img_name in os.listdir(ImageTypePaths.RAW.value):

        # Create the label image object
        label_img = LabelImage(img_name, model, qr_detector=qr_detector, ocr=reader)

        # Apply filters such that the image is ready to be processed (grayscaled and blurred)
        label_img.pre_process_image()

        # Apply a mask (adaptive threshold) to the image and save the result
        label_img.mask_image()
        label_img.save()

        # Detect the contours and corners for the label and save the end result (sketch of contour and corners)
        label_img.detect_contours()
        label_img.detect_corners()
        label_img.save()

        # Apply perspective transform according to the calculated corners and save the result
        label_img.perspective_transform()

        # If there is an error detecting the qr code, detect corners with only maximizing 4 directions
        if(label_img.detect_qrcode()[1] is None):
            label_img.detect_corners(use_4_direction=True)
            label_img.save()

            label_img.perspective_transform()
            if(label_img.detect_qrcode()[1] is None):
                raise Exception("QR code cannot be detected")

        label_img.correct_orientation()
        label_img.save()

        # (Optional) Apply canny edge detection to the perspective transformed image and save the result
        # label_img.detect_edge()
        # label_img.save()

        # Divide the image into segments where each corresponds to a field on the label
        label_img.divide()
        label_img.save()

        label_img.detect_text()
        label_img.detect_text(correction="full")

        label_img.print_ascii_label()
        print(img_name)

if __name__ == '__main__':
    main()