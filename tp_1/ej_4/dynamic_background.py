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
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #cv.imshow('frame', frame[:,:,1])
        #hist = cv.calcHist([frame[:,:,1]], [0], None, [256], [0, 256])
        h,w,c = frame.shape
        bg = background[:h,:w,:]
        green_channel = frame[:,:,1]
        mask_bool = (green_channel == 255)
        mask = np.uint8(mask_bool)
        mask *= 255
        background_img = cv.bitwise_and(bg,bg,mask=mask)
        mask_inv = cv.bitwise_not(mask)
        person = cv.bitwise_and(frame,frame,mask=mask_inv)
        #person = cv.GaussianBlur(person, (5,5),0, borderType = cv.BORDER_DEFAULT)
        result_image = cv.add(person,background_img)
        #print(frame[mask].shape)
        #print(bg[:,:,0][mask].shape)
        #print(cv.threshold(green_channel, 255, 255, cv.THRESH_BINARY+cv.THRESH_OTSU))
        #blur = cv.GaussianBlur(mask*255, (0,0), sigmaX=5, sigmaY=5, borderType = cv.BORDER_DEFAULT)

        # how to filter using mask? I want to use mask to make a new image with
        # some pixels of the background image and some pixels of the current frame
        #mask = cv.GaussianBlur(mask, (0,0), sigmaX=5, sigmaY=5, borderType = cv.BORDER_DEFAULT)

        #for i in range(3):
        #    new_channel = np.zeros((h,w), dtype=np.uint8)
        #    new_channel[mask] = bg[:,:,i][mask]
        #    new_channel[np.logical_not(mask)] = frame[:,:,i][np.logical_not(mask)]
        #    result_image[:,:,i] = new_channel
        
        
        
        cv.imshow('frame', result_image)
        #plt.hist(green.flatten(),256,[0,256], color = 'r')
        #plt.xlim([0,256])
        #plt.show()

        if cv.waitKey(10) == ord('q'):
            break   
    cap.release()
    cv.destroyAllWindows()