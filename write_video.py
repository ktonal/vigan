# make sure you have the codec
# sudo apt-get install ffmpeg x264 libx264-dev

import cv2
import numpy as np
from mimikit import FileWalker
import re

def sorted_image_list(input_folder):
    files = list(FileWalker('img', input_folder))
    files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    return files


class VideoGen():

    def __init__(self, fps=15, codec='mp4v'):
        self.codec = codec
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc(*codec)

    def write(self, image_stream, outfile):
        # get first image to know the shape
        image_iterator = iter(image_stream)
        filename = next(image_iterator)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        writer = cv2.VideoWriter(outfile, self.fourcc, self.fps, (width, height))
        writer.write(img)
        # write remaining images
        for filename in image_iterator:
            img = cv2.imread(filename)
            writer.write(img)
        writer.release()


# to write a folder of images to a video file in numerically sorted order do:
#
#
# vg = VideoGen(codec='H265')
# vg.write(sorted_image_list('images_output2'), 'testvideo.mp4')

