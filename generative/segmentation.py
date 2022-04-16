import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import imutils

texture_data_dir = os.getcwd() + '/texturedata/'
texture_dir = os.getcwd() + '/block/'


#preproc input img
img_path = 'water.jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print("img shape: ")
print(img.shape)
pixel_vals = img.reshape((-1,3))
pixel_vals = np.float32(pixel_vals)
pixels_arr = pixel_vals.flatten()

print("pixel_vals shape: ")
print(pixel_vals.shape)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
k = 1
attempts = 10

##segmentation
ret, labels, centers = cv2.kmeans(pixel_vals,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
print("labels shape: ")
print(labels.shape)
centers = np.uint8(centers)

segmented_data = centers[labels.flatten()]
segmented_img = segmented_data.reshape(img.shape)

print("ret: " + str(ret))
print("centers: " + str(centers))
plt.imshow(segmented_img)
plt.show(block=False)

print(segmented_img.shape)

##split channels and compute RGB histogram
chans = cv2.split(img)

print("chans: ")
print(chans[0].shape)
colors = ("r", "g", "b")

for i in range(k):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    axs = (ax1, ax2, ax3)
    plt.title("segment " + str(i))
    plt.xlabel("bins")
    plt.ylabel("# pix")

    segment_chans = list(map(lambda chan: chan.flatten()[labels.flatten()==i], chans))

    print("segment_chans shape: ")
    print(segment_chans[0].shape)

    histRGB = cv2.calcHist([img], [0, 1, 2], None, [8,8,8], [0,256, 0,256, 0,256])
##load in mc texture histograms

compare_results = {}

for root, dirs, files in os.walk(texture_data_dir):
    for file in files:

        mcblock = os.path.splitext(file)[0]

        resultRGB = 0
        mcRGBhist_reshaped = np.loadtxt(os.path.join(root, file))
        mcRGBhist = mcRGBhist_reshaped.reshape(8,8,8)
        imghist_norm = np.float32(cv2.normalize(imghist, imghist).flatten())
        mchist_norm = np.float32(cv2.normalize(mchist, mchist).flatten())
        resultRGB += cv2.compareHist(imghist_norm, mchist_norm, method)
        compare_results[mcblock] = resultRGB


compare_results_sorted = sorted(compare_results.items(), key=lambda x:x[1], reverse=True)
print(compare_results_sorted)

#display top 5 closest blocks for each segment
for i in range(k):
    for j in range(40):
        sim_mcblock = compare_results_sorted[j][0]
        img = cv2.imread(os.path.join(texture_dir, sim_mcblock + '.png'))
        resized = imutils.resize(img, width=200)
        cv2.imshow(sim_mcblock, resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
