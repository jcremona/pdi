import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='''
    Set dynamic background
    ''')
    parser.add_argument('video_path', help='Path to original video')
    parser.add_argument('video_background', help='Path to background video')
    
    args = parser.parse_args()    
    video_person = cv.VideoCapture(args.video_path)
    video_background = cv.VideoCapture(args.video_background)
    while video_person.isOpened():
        ret_p, frame = video_person.read()
        ret_b, background = video_background.read()
        # if frame is read correctly ret is True
        if not ret_p:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        h,w,c = frame.shape
        bg = background[:h,:w,:]

        # Build an appropiate mask to cut just the person
        # We discard the pixels where the difference between 
        # green channel and blue channel (and red channel) is too large
        green_blue_diff = cv.subtract(frame[:,:,1],frame[:,:,0])#(frame[:,:,1] / np.log(frame[:,:,0]+0.1)).astype('uint8')
        green_red_diff = cv.subtract(frame[:,:,1],frame[:,:,2])#(frame[:,:,1] / np.log(frame[:,:,2]+0.1)).astype('uint8')        
        _,thresh1 = cv.threshold(green_blue_diff,14,255,cv.THRESH_BINARY_INV)
        _,thresh2 = cv.threshold(green_red_diff,14,255,cv.THRESH_BINARY_INV)

        # add both masks
        mask = cv.add(thresh1,thresh2)
        
        # Smooth the mask
        mask = cv.medianBlur(mask, 3)

        # The inverse mask will be used over the new background
        mask_inv = cv.bitwise_not(mask)

        # We finally replace the green background with a new video frame
        # Make place for the person in the new background
        background_img = cv.bitwise_and(bg,bg,mask=mask_inv)
        # We obtain the person from the original frame
        person = cv.bitwise_and(frame,frame,mask=mask)
        # We paste both the person and the new background
        result_image = cv.add(person, background_img)
        
        cv.imshow('frame',result_image)     
    
        if cv.waitKey(15) == ord('q'):
            break   
    video_person.release()
    video_background.release()
    cv.destroyAllWindows()