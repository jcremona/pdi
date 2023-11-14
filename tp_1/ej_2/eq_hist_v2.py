import cv2
import numpy as np

def custom_equalization(roi):
    # Both implementations differ because cv2.equalizeHist implements:
    # > cdf_m = np.ma.masked_equal(cdf,0) # Remove zeros
    # if min != max then the minimum will be 0 and maximum will be 255
    # > cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min()) 
    # > cdf = np.ma.filled(cdf_m,0).astype('uint8')

    # Local histogram
    hist = cv2.calcHist([roi], [0], None, [256], [0, 256])

    # Equalization
    num_elements = roi.shape[0] * roi.shape[1]
    hist_norm = hist / num_elements
    cdf = hist_norm.cumsum()

    eq_roi = (cdf[roi] * 255).astype(np.uint8)
    
    return eq_roi

def opencv_equalization(roi):
    return cv2.equalizeHist(roi)

def local_histogram_equalization(image, M, N):
    
    rows, cols = image.shape[:2]
    half_win_i = M // 2
    half_win_j = N // 2
    result_image = np.copy(image)
    iter = 0
    for i in range(half_win_i, rows - half_win_i):
        for j in range(half_win_j, cols - half_win_j):
            # Define roi
            roi = image[i - half_win_i: i + half_win_i + 1, j - half_win_j: j + half_win_j + 1]
            eq_roi = opencv_equalization(roi)            
            result_image[i, j] = eq_roi[half_win_i, half_win_j]

    return result_image


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='''
    Local histogram equalization
    ''')
    parser.add_argument('image_path', help='Path to image')
    parser.add_argument('--window_size', type=int, nargs=2, help='Angle for motion kernel (in degrees)')
    args = parser.parse_args()

    input_image = cv2.imread(args.image_path, 0)     
    input_image = cv2.GaussianBlur(input_image, (3,3),0)     

    # Set background to white
    mask = (input_image > 200)
    input_image[mask] = 255

    output_image = local_histogram_equalization(input_image, args.window_size[0], args.window_size[1])

    # Remove salt&pepper noise (median filter)
    #output_image = cv2.medianBlur(output_image, 3)
    _,output_image = cv2.threshold(output_image, 187, 255, cv2.THRESH_BINARY)
    cv2.imwrite('eq_img.jpg', output_image)
    cv2.imshow('Fixed', output_image)
            
    # Wait for a key press and then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()   
