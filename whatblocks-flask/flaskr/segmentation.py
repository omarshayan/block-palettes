import os
import numpy as np
import pandas as pd
import cv2
import imutils
import sys
from PIL import Image


def extractPalette(img_path, img_file, seg_path):

    num_results = 5

    img_fullpath = os.path.join(img_path, img_file)
    texture_data_dir = os.getcwd() + '/texturedata/'
    texture_dir = os.getcwd() + '/block/'

    print(img_fullpath, file=sys.stdout)
    print(texture_data_dir, file=sys.stdout)
    print(texture_dir, file=sys.stdout)
    
    #preproc input img
    img = cv2.imread(img_fullpath)
    print("img type: " + str(type(img)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

    pixel_vals = img.reshape((-1,3))
    pixel_vals = np.float32(pixel_vals)
    pixels_arr = pixel_vals.flatten()


    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 5
    attempts = 10

    ##segmentation
    ret, labels, centers = cv2.kmeans(pixel_vals,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

    centers = np.uint8(centers)

    segmented_data = centers[labels.flatten()]
    segmented_img = segmented_data.reshape(img.shape)

    print(type(segmented_img), file=sys.stdout)
    print(segmented_img.shape, file=sys.stdout)

    ##split channels and compute RGB histogram
    chans = cv2.split(img)
    colors = ("r", "g", "b")

    seg_results = []

    ##for each segment
    for i in range(k):
        
        
        segment_chans = list(map(lambda chan: chan.flatten()[labels.flatten()==i], chans))
        seg_chans_img = np.transpose(np.asarray(segment_chans)).reshape(len(segment_chans[0]), 1, 3)
        print("seg chans")
        print(type(seg_chans_img), file=sys.stdout)
        print(seg_chans_img.shape, file=sys.stdout)
        imghistRGB = cv2.calcHist([seg_chans_img], [0, 1, 2], None, [8,8,8], [0,256, 0,256, 0,256]) ##calculate 3d histogram for segment

        
        ##loop through mc texture histograms
        compare_results = {}

        for root, dirs, files in os.walk(texture_data_dir):
            for file in files:

                mcblock = os.path.splitext(file)[0]
                
                resultRGB = 0
                mcRGBhist_reshaped = np.loadtxt(os.path.join(root, file))
                mcRGBhist = mcRGBhist_reshaped.reshape(8,8,8)
                imghist_norm = np.float32(cv2.normalize(imghistRGB, imghistRGB).flatten())
                mchist_norm = np.float32(cv2.normalize(mcRGBhist, mcRGBhist).flatten())
                method = cv2.HISTCMP_CORREL
                resultRGB += cv2.compareHist(imghist_norm, mchist_norm, method)
                compare_results[mcblock] = resultRGB



        seg_results.append(sorted(compare_results.items(), key=lambda x:x[1], reverse=True)[0:num_results])
##        print(compare_results_sorted, file=sys.stdout)

    seg_image = Image.fromarray(segmented_img)

    seg_filename = 'seg_' + img_file
    seg_path = os.path.join(seg_path, seg_filename)
    seg_image.save(seg_path)
    return seg_results, seg_filename



