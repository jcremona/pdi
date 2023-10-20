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
        mask = (green_channel > 254)
        result_image = np.zeros(frame.shape, dtype=np.uint8)
        
        # how to filter using mask? I want to use mask to make a new image with
        # some pixels of the background image and some pixels of the current frame

        for i in range(3):
            new_channel = np.zeros((h,w), dtype=np.uint8)
            new_channel[mask,i] = bg[bg[:,:,i]==mask]
            new_channel[np.logical_not(mask),i] = frame[np.logical_not(mask),i]
            result_image[:,:,i] = new_channel
        #cv.imwrite('g.png', im_th)
        
        cv.imshow('frame', frame)
        #plt.hist(green.flatten(),256,[0,256], color = 'r')
        #plt.xlim([0,256])
        #plt.show()

        if cv.waitKey(10) == ord('q'):
            break   
    cap.release()
    cv.destroyAllWindows()