#!/usr/bin/env python3
import cv2
from scipy.linalg import toeplitz
import numpy as np
from numpy.linalg import pinv
import time

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='''
    This script takes two data files with timestamps and associates them   
    ''')
    parser.add_argument('image_path', help='Path to image')
    args = parser.parse_args()
    # Load an RGB image
    image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    for M in range(1,40):
        for N in range(1,40):
            print(f"{M}, {N}")
            col = [1/M if i < M else 0 for i in range(image.shape[0])]
            row = [1/M if i == 0 else 0 for i in range(image.shape[0])]
            H_ver = toeplitz(col, row)
            H_ver = np.array(H_ver)
            H_ver_inv = np.linalg.inv(H_ver)

            col = [1/N if i < N else 0 for i in range(image.shape[1])]
            row = [1/N if i == 0 else 0 for i in range(image.shape[1])]
            H_hor = toeplitz(col, row)
            H_hor = np.array(H_hor)
            H_hor_inv = np.linalg.inv(H_hor.transpose())            

            fixed = np.dot(H_ver_inv, np.dot(image, H_hor_inv)) #, dtype = cv2.CV_32F)
            fixed = cv2.normalize(fixed, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            # Display the image
            cv2.imshow('Original', image)
            cv2.imshow('Fixed', fixed)
            
            # Wait for a key press and then close the window
            cv2.waitKey(100)

    
    cv2.destroyAllWindows()    

