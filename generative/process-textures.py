import os 
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load mc textures
textures_dir = os.getcwd() + '/block'
texture_data_dir = os.getcwd() + '/texturedata/'
if not os.path.exists(texture_data_dir):
    os.makedirs(texture_data_dir)
print("getting textures from " + textures_dir)
textures = []
for root, dirs, files in os.walk(textures_dir):
    for file in files:
        print(file)
        img = cv2.imread(os.path.join(root, file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        chans = cv2.split(img)
        colors = ("r", "g", "b")

##        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
##        axs = (ax1, ax2, ax3)
##        plt.title("rgb hists")
##        plt.xlabel("bins")
##        plt.ylabel("# pix")

        histRGB = {}
##        for (chan, color, ax) in zip(chans, colors, axs):
        for (chan, color) in zip(chans, colors):
            hist = cv2.calcHist([chan], [0], None, [256], [0,256])
##            ax.plot(hist, color=color)
##            plt.xlim([0, 256])
            histRGB[color]=hist.flatten()

    
        
        df = pd.DataFrame(data=histRGB)
        df.to_csv(os.path.join(texture_data_dir, os.path.splitext(file)[0]) + ".csv")
        
