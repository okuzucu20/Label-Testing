#!/usr/bin/env python
from paddleocr import PaddleOCR
from src.testers.THC100D_Tester import THC100DTester
from src.testers.OVI_Tester import OVI_Tester
import cv2 as cv

def main():
    # Initialize qr detector and ocr
    qr_detector = cv.QRCodeDetector()
    reader = PaddleOCR(lang='en')
    #THC100DTester(qr_detector, reader).test_label()
    OVI_Tester(reader).test_indicator()
    

if __name__ == '__main__':
    main()