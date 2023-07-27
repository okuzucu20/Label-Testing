import cv2 as cv
import os
import numpy as np
import qr2text
from paddleocr import PaddleOCR
from src.devimg.DeviceImage import DeviceImage
from src.utils import *
from src.constants import *

QR_SEGMENT_INDICES = {Models.THC100D: 4}
CORRECTION_INDICES = {Models.THC100D: {0: None, 5: "key"}} # keys indicate segment indices
                                                           # if value is None, change full text
                                                           # elif value is "key" change the part before ":" in the text (if found)
                                                           # elif value is "val" change the part after  ":" in the text (if found)
#GROUND_TRUTH = {Models.THC100D: []}

class LabelImage(DeviceImage):

    def __init__(self, img_filename: str, model: Models, qr_detector, ocr) -> None:
        
        super().__init__(img_filename, model, test_type=TestTypes.LABEL)

        # QR Detector and OCR are given as arguments in order to eliminate multiple 
        # initialization of these classes for different images
        self.qr_detector = qr_detector
        self.ocr = ocr

        # Transformed images
        self.img_canny = None
        self.img_segment = None

        # Label segment information
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
    
    def detect_qrcode(self):

        if self.img_perspective is None:
            raise Exception("Perspective transformed image is not calculated before qrcode detection")
        img = self.img_perspective

        self.qrcode_info = self.qr_detector.detectAndDecode(img)
        return self.qrcode_info
    
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
    
    def divide(self, img=None):

        if img is None:
            if self.img_perspective is None:
                raise Exception("Perspective transformed image is not calculated before dividing the label into segments")
            img = self.img_perspective

        # Pre-process and mask the given image
        self.mask(img=self.pre_process(img=img))

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
            return 255 - self.mask(img=self.pre_process(img=img))

        for i, bbox in segments:

            if i == QR_SEGMENT_INDICES[self.model]:
                continue

            # Crop image to see only the current segment
            cropped_img = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

            # Pre-process and mask the cropped image
            cropped_img = 255 - self.mask(img=cropped_img)

            # Read the text with ocr
            text_list = self.ocr.ocr(cv.cvtColor(cropped_img, cv.COLOR_BGR2RGB))[0] # x_ths=100, y_ths=100, width_ths=100, height_ths=100, ycenter_ths=100

            # Sort the texts in the list according to their individual bbox locations (middle x coordinate)
            text_list_sorted = sorted(text_list, key=(lambda text_info: (text_info[0][0][0] + text_info[0][2][0])/2))

            text_merged = ""; score_mean=0; j=0
            for text_info in text_list_sorted:
                bb, text, score = text_info[0], text_info[1][0], text_info[1][1]
                j+=1

                # Update the mean score for the segment and merge the text
                score_mean += score
                text_merged += text + " "
            
            if correction != "none" and score_mean < self.segment_text_scores[i]:
                continue

            # Update the segment values
            self.segment_text_list[i] = text_list_sorted
            self.segment_text_merged[i] = text_merged[:-1]
            self.segment_text_scores[i] = score_mean
    
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
            
    @staticmethod
    def print_ascii_label_thc100d(qrcode_info, segment_text_merged):
        label_length = 80
        qr_one_line_len = 22
        print(qrcode_info[0])
        qr_ascii = qr2text.QR(2).from_text(qrcode_info[0]).to_ascii_art(trim=True)

        print("-" * label_length)
        print(f"|" + " "*(label_length -2) + "|")
        print(f"|   {segment_text_merged[0]}" + " "*(label_length - len(segment_text_merged[0]) -5) + "|")
        print(f"|" + " "*(label_length -2) + "|")

        print("-" * label_length)
        print(f"|   " + " "*(label_length//2 -5) + "|" + " "*(label_length//2 -1) + "|")
        print(f"|   {segment_text_merged[1]}" + " "*(label_length//2 - len(segment_text_merged[1]) -5) + "|", end="")
        print(f"   {segment_text_merged[2]}" + " "*(label_length//2 - len(segment_text_merged[2]) -4) + "|")
        print(f"|   " + " "*(label_length//2 -5) + "|" + " "*(label_length//2 -1) + "|")

        print("-" * label_length)
        #print(f"|   " + " "*(label_length - qr_one_line_len -9) + "|" + " "*(qr_one_line_len+3) + "|")
        print(f"|   " + " "*(label_length - qr_one_line_len -9) + "|", end="")
        print(f" ", qr_ascii[:qr_one_line_len-1], " |")
        print(f"|   {segment_text_merged[3]}" + " "*(label_length - len(segment_text_merged[3]) - qr_one_line_len -9) + "|", end="")
        print(f" ", qr_ascii[qr_one_line_len:2*qr_one_line_len-1], " |")
        print(f"|   " + " "*(label_length - qr_one_line_len -9) + "|", end="")
        print(f" ", qr_ascii[2*qr_one_line_len:3*qr_one_line_len-1], " |")

        print("-" * (label_length - qr_one_line_len -5), end="|")
        print(f" ", qr_ascii[3*qr_one_line_len:4*qr_one_line_len-1], " |")

        #print(f"|   " + " "*(label_length - qr_one_line_len -9) + "|" + " "*(qr_one_line_len+3) + "|")
        print(f"|   " + " "*(label_length - qr_one_line_len -9) + "|", end="")
        print(f" ", qr_ascii[4*qr_one_line_len:5*qr_one_line_len-1], " |")
        print(f"|   {segment_text_merged[5]}" + " "*(label_length - len(segment_text_merged[5]) - qr_one_line_len -9) + "|", end="")
        print(f" ", qr_ascii[5*qr_one_line_len:6*qr_one_line_len-1], " |")
        print(f"|   " + " "*(label_length - qr_one_line_len -9) + "|", end="")
        print(f" ", qr_ascii[6*qr_one_line_len:7*qr_one_line_len-1], " |")

        print("-" * (label_length - qr_one_line_len -5), end="|")
        print(f" ", qr_ascii[7*qr_one_line_len:8*qr_one_line_len-1], " |")

        #print(f"|   " + " "*(label_length - qr_one_line_len -9) + "|" + " "*(qr_one_line_len+3) + "|")
        print(f"|   " + " "*(label_length - qr_one_line_len -9) + "|", end="")
        print(f" ", qr_ascii[8*qr_one_line_len:9*qr_one_line_len-1], " |")
        print(f"|   {segment_text_merged[6]}" + " "*(label_length - len(segment_text_merged[6]) - qr_one_line_len -9) + "|", end="")
        print(f" ", qr_ascii[9*qr_one_line_len:10*qr_one_line_len-1], " |")
        print(f"|   " + " "*(label_length - qr_one_line_len -9) + "|", end="")
        print(f" ", qr_ascii[10*qr_one_line_len:11*qr_one_line_len-1], " |")

        print("-" * label_length)
        print(f"|" + " "*(label_length -2) + "|")
        print(f"|   " + str(segment_text_merged[7]).center(label_length-10) + " " * 5 + "|")
        print(f"|" + " "*(label_length -2) + "|")
        print("-" * label_length)

    def print_ascii_label(self):
        if self.model == Models.THC100D:
            LabelImage.print_ascii_label_thc100d(self.qrcode_info, self.segment_text_merged)

    #@staticmethod
    #def test_path_label():
    #    return os.path.join(IMG_FOLDER, Models.THC100D.value, TestTypes.LABEL.value)

    #@property
    #def test_path(self):
    #    return os.path.join(IMG_FOLDER, self.model, self.test_type.value)

