import os
from src.devimg.IndicatorImage import IndicatorImage
from src.constants import *

import cv2 as cv
import matplotlib.pyplot as plt

class OVI_Tester:

    def __init__(self, ocr) -> None:
        self.ocr = ocr
        self.model = Models.OVI

    def test_indicator_image(self, img: IndicatorImage):

        #img.detect_text()
        #img.get_power_indicator_info()
        print(img.img_name)
        print(img.detect_text())
        print(img.check_power())

        plt.imshow(img.display_search_area())
        plt.show()
        
        img.detect_edge()
        img.save()

        img.mask()
        img.save()



    def test_indicator(self):

        for img_name in os.listdir(os.path.join(INDICATOR_PATH(self.model.value), ImageTypes.RAW.value)):

            # Test every image inside raw folder
            self.test_indicator_image(IndicatorImage(img_name, self.model, self.ocr))

def main():
    pass

if __name__ == '__main__':
    main()