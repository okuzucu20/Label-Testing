from src.devimg.LabelImage import LabelImage
from src.constants import *
import cv2 as cv
import os

class THC100DTester:

    def __init__(self, qr_detector, ocr) -> None:
        self.qr_detector = qr_detector
        self.ocr = ocr
        self.model = Models.THC100D

    def test_label_image(self, img: LabelImage):

        # Apply filters such that the image is ready to be processed (grayscaled and blurred)
        img.pre_process()

        # Apply a mask (adaptive threshold) to the image and save the result
        img.mask()
        img.save()

        # Detect the contours and corners for the label and save the end result (sketch of contour and corners)
        img.detect_contours()
        img.detect_corners()
        img.save()

        # Apply perspective transform according to the calculated corners and save the result
        img.perspective_transform()

        # If there is an error detecting the qr code, detect corners with only maximizing 4 directions
        if(img.detect_qrcode()[1] is None):
            img.detect_corners(use_4_direction=True)
            img.save()

            img.perspective_transform()
            if(img.detect_qrcode()[1] is None):
                raise Exception("QR code cannot be detected")

        img.correct_orientation()
        img.save()

        # (Optional) Apply canny edge detection to the perspective transformed image and save the result
        # img.detect_edge()
        # img.save()

        # Divide the image into segments where each corresponds to a field on the label
        img.divide()
        img.save()

        img.detect_text()
        img.detect_text(correction="full")

        img.print_ascii_label()

    def test_label(self):

        for img_name in os.listdir(os.path.join(LABEL_PATH(self.model.value), ImageTypes.RAW.value)):

            # Test every image inside raw folder
            self.test_label_image(LabelImage(img_name, self.model, self.qr_detector, self.ocr))

def main():
    pass

if __name__ == '__main__':
    main()