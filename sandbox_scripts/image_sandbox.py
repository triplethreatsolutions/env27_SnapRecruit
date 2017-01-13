import cv2
from PIL import Image, ImageFilter, ImageDraw
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

cv2_image_location = '..\\..\\images\\usjn_mod1.png'
pil_image_location = '..\\..\\images\\usjn_mod1.png'


if __name__ == '__main__':
    try:
        cv2_img = cv2.imread(cv2_image_location)
    except IOError:
        print(' Unable to open image: ', cv2_image_location)

    try:
        pil_image = Image.open(pil_image_location)
    except IOError:
        print(' Unable to open image: ', pil_image_location)

    pil_img_sharpen = pil_image.filter(ImageFilter.SHARPEN)

    img_grey = cv2.cvtColor(np.array(pil_img_sharpen), cv2.COLOR_BGR2GRAY)

    img_denoise = cv2.fastNlMeansDenoising(img_grey)

    #img_bilateral = cv2.bilateralFilter(img_denoise, 3, 50, 50)

    threshold = cv2.adaptiveThreshold(img_denoise, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2)

    cv2.imshow('Original', cv2_img)
    cv2.imshow('Sharpen', img_grey)
    cv2.imshow('Denoise', img_denoise)
    #cv2.imshow('img_bilateral', img_bilateral)
    cv2.imshow('Adaptive Threshold', threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(img_grey)

#Open and initialise an image object, Convert('L') it gray scale
#pil_im = array(Image.open("..\\images\\usjn_mod1.png").convert('L'))
#print (pil_im)
# create a new figure
#figure()
# show contours with origin upper left corner

#Turn interactive mode on
#matplotlib.pyplot.ion()
#read image into a array
#img = mpimg.imread("..\\images\\first.png")
#print (img)
#imgplot = plt.imshow(img)


#imshow(pil_im)
#Display image to user
#pil_im.show()


# read image to array
#im = array(Image.open("images\\usjn.png").convert('L'))
# plot the image
# plot the image
# imshow(im)
# print ("Please click 3 points")
# x = ginput(4)
# print ("you clicked: %d", x)
# box = (x[0],x[1],x[2],x[3])
# print (box)
# region = im.crop(box)
# region.save("images\\cropped.jpg")
# plot the points with red star-markers
#plot(x, y, 'r*')
# line plot connecting the first two points
#plot(x[:2], y[:2])
# add title and show the plot
# title("Plotting: images\\usjn_mod.png")
# show()
