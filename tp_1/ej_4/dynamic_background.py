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
    #parser.add_argument('--window_size', type=int, nargs=2, help='Angle for motion kernel (in degrees)')
    args = parser.parse_args()    
    cap = cv.VideoCapture(args.video_path)
    background = cv.imread(args.video_background)
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        h,w,c = frame.shape
        bg = background[:h,:w,:]
        # green_channel = frame[:,:,1]
        # mask_bool = (green_channel != 255)
        # mask = np.uint8(mask_bool)
        # mask *= 255        
        green_blue_ratio = (frame[:,:,1] / (frame[:,:,0]+0.1)).astype('uint8')
        green_red_ratio = (frame[:,:,1] / (frame[:,:,2]+0.1)).astype('uint8')
        # FIXME so many values are near 1 because 255 divided by a similar number is near 1 
        # Could be solved applying a log to the divisor? or something similar
        _,thresh1 = cv.threshold(green_blue_ratio,1,255,cv.THRESH_BINARY_INV)
        _,thresh2 = cv.threshold(green_red_ratio,1,255,cv.THRESH_BINARY_INV)
        # FIXME perhaps it could be improved using the former mask ((green_channel != 255))
        mask = cv.add(thresh1,thresh2)
        #mask = cv.subtract(mask, cv.bitwise_not(cm))        
        mask_inv = cv.bitwise_not(mask)
        background_img = cv.bitwise_and(bg,bg,mask=mask_inv)
        person = cv.bitwise_and(frame,frame,mask=mask)

        # Blur the walking man
        blurred_person = cv.GaussianBlur(person, (3, 3), 0)
        #sobel = cv.Sobel(src=mask, ddepth=cv.CV_8U, dx=1, dy=1, ksize=1)
        #sobel = np.stack((sobel,)*3, axis=-1)
        #canny = cv.Canny(image=mask, threshold1=0, threshold2=10)
        #canny = np.stack((canny,)*3, axis=-1)
        
        # Find the contour
        contours, hierarchy = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Obtain a contour mask
        contour_mask = np.zeros(frame.shape, np.uint8)
        cv.drawContours(contour_mask, contours, -1, (255,255,255),5)        

        # Replace the contour of the person with the blurred contour (using the contour mask)
        output = np.where(contour_mask==np.array([255, 255, 255]), blurred_person, person)

        result_image = cv.add(output, background_img)
        cv.imshow('frame',result_image)     

        if cv.waitKey(10) == ord('q'):
            break   
    cap.release()
    cv.destroyAllWindows()