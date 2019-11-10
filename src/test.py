from images.image import Image
from images.image_type import ImageType

import cv2
from PIL import Image as PILImage
from pprint import pprint


def printcached(img):
    print("Cached results: ")
    pprint({key: img_pil.img[key] is not None for key in img_pil.img.keys()})


img_pil = Image(PILImage.fromarray(cv2.imread('../img/object_detection_yolo_mxnet.png')))

printcached(img_pil)
img_pil.asnumpy();  printcached(img_pil)
img_pil.asopencv(); printcached(img_pil)
img_pil.asnumpy();  printcached(img_pil)
img_pil.asopencv(); printcached(img_pil)
img_pil.aspil();    printcached(img_pil)


