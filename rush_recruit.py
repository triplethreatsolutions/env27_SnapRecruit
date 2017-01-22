# rush_recruit
import csv
import glob

import pyocr.builders
from PIL import Image
from pylab import *

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

# initialize recruit info list
recruit_info = list()

folders = glob.glob('static/images/*/')
#print (folders)

print(' ')

output = list()
for folder in folders:
    #print (folder)
    img_files = glob.glob(folder+'/*.png')
    #print (img_files)

    for im in img_files:
        try:
            print('Getting image to extract data: ' + str(im))
            pil_image = Image.open(im)
        except IOError:
            print(' Unable to open image: ' + str(im))

        print('Running tesseract OCR...')
        row_info = tool.image_to_string(
            pil_image,
            lang=lang,
            builder=pyocr.builders.TextBuilder()
        )
        row_data = row_info.encode('utf-8', 'replace')
        output.append(row_data)
        print (output)

    # Load all each individual file data into list
    recruit_info.append(output)
    output = list()
    print(' ')

    # Open a file and write csv
    print('Writing data to CSV file...')
    with open('snap_results.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, dialect='excel')
        for item in recruit_info:
            writer.writerow(item)

print('Image data extraction completed!')



