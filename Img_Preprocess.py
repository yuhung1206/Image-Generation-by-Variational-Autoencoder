# import module
import numpy as np
import os
import cv2
from itertools import chain

ImgSize = 32  # for resize

# change directory
os.chdir(r"C:\...\Project")  # alter the directory !!!
ImagePath = "data"
StorePath = "./Processed_Img"

# construct the directory if the directory not exist
if not os.path.isdir(StorePath):
    os.mkdir(StorePath)


# function to Read All images [input : directory name]
def read_All_File_Directory(DirectoryName):
    Img = []  # store all the Img in this directory

    for filename in os.listdir(r"./" + DirectoryName):
        #print(filename) #just for test
        #img is used to store the image data
        img = cv2.imread("./" + DirectoryName + "/" + filename)
        # Resize the Image -> Cubic Method
        resizeImg = cv2.resize(img, (ImgSize, ImgSize), interpolation=cv2.INTER_CUBIC)
        # reshape 2D --> 1D image
        tmp = resizeImg.reshape(ImgSize*ImgSize, 3)  # [1024 X 3]
        tmp2 = list(chain.from_iterable(zip(tmp[:,0], tmp[:,1], tmp[:,2])))  # [4096]
        # Store the Image
        Img.append(tmp2)
        print(filename)
        
    return Img

Img_Total = read_All_File_Directory(ImagePath)
# Transform to numpy array
Img_Total = np.array(Img_Total)  # [21551, 3072]
Img_Total = Img_Total/255.  # Normalize to (0,1)
# Save to Numpy
np.save(StorePath + "/Img_merge"+str(ImgSize)+".npy", Img_Total)
