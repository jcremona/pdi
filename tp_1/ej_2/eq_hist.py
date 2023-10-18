import cv2
import numpy as np

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

            # Local histogram
            hist = cv2.calcHist([roi], [0], None, [256], [0, 256])

            # Equalization
            num_elements = roi.shape[0] * roi.shape[1]
            hist_norm = hist / num_elements
            cdf = hist_norm.cumsum()

            eq_roi = (cdf[roi] * 255).astype(np.uint8)            
            print("----")
            print(roi)
            print(eq_roi)            
            print(cv2.equalizeHist(roi))
            
            #print(cv2.calcHist([eq_roi], [0], None, [256], [0, 256]))
            #print(cv2.calcHist([cv2.equalizeHist(roi)], [0], None, [256], [0, 256]))

            result_image[i, j] = eq_roi[half_win_i, half_win_j]
            iter +=1 
            if iter == 200:
                exit(-1)

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

    output_image = local_histogram_equalization(input_image, args.window_size[0], args.window_size[1])

    # Optional, remove salt&pepper noise (median filter)
    #output_image = cv2.medianBlur(output_image, 3)

    cv2.imwrite('eq_img.jpg', output_image)
