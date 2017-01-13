# snap
import codecs
from PIL import Image, ImageFilter, ImageDraw
import sys
import os
import csv
import pyocr.builders
from itertools import islice, chain, repeat
from pylab import *
import numpy as np
import cv2


_no_padding = object()

tools = pyocr.get_available_tools()
if 0 == len(tools):
    print("No OCR tool found")
    sys.exit(1)
# The tools are returned in the recommended order of usage
tool = tools[0]
print("Will use tool '%s'" % (tool.get_name()))
# Ex: Will use tool 'libtesseract'

langs = tool.get_available_languages()
print("Available languages: %s" % ", ".join(langs))
lang = langs[0]
print("Will use lang '%s'" % (lang))
# Ex: Will use lang 'fra'

try:
    pil_image = Image.open('..\\static\images\_3.png')
except IOError:
    print(' Unable to open image: ')

#pil_img_sharpen = pil_image.filter(ImageFilter.SHARPEN)

#img_grey = cv2.cvtColor(np.array(pil_img_sharpen), cv2.COLOR_BGR2GRAY)

#img_denoise = cv2.fastNlMeansDenoising(img_grey)
#img_bilateral = cv2.bilateralFilter(img_grey, 9, 75, 75)
#threshold = cv2.adaptiveThreshold(img_denoise, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 31)

#blur = cv2.GaussianBlur(img_grey, (3, 3), 0)
#ret3, th3 = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Convert a cv2 image to a PIL image
#pil_im = Image.fromarray(th3)

listit = tool.image_to_string(
    pil_image,
    lang=lang,
    builder=pyocr.builders.TextBuilder()
)
listit = listit.encode('utf-8', 'replace')
num_list = listit.split()
print (num_list)

# Open a file and write csv
with open('snap_results.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, dialect='excel')
    #for item in num_list:
        #print (item)
    writer.writerow(num_list)



