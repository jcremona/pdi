import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='''
    Removing periodic noise
    ''')
    parser.add_argument('image_path', help='Path to image')
    #parser.add_argument('--window_size', type=int, nargs=2, help='Angle for motion kernel (in degrees)')
    args = parser.parse_args()
    
    image = cv2.imread(args.image_path)
    # OpenCV uses BGR as its default colour order for images, matplotlib uses RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    result_image = np.zeros(image.shape, dtype=np.uint8)
    print(result_image.dtype)
    print(result_image.shape)
    for i in range(3):
        rgb_channel = image[:, :, i]
        F = np.fft.fft2(rgb_channel)
        # S = np.abs(F) # spectrum
        # Slog = np.log(1.0 + S) # enhance visualization
        
        FF = np.fft.fftshift(F)
        FF[184, 224] = 0
        FF[176, 240] = 0
        FF[208, 272] = 0
        FF[200, 288] = 0

        result_image[:,:,i] = np.real(np.fft.ifft2(np.fft.ifftshift(FF)))
    
    plt.subplot(121), plt.imshow(image), plt.title('Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(result_image), plt.xticks([]), plt.yticks([]), plt.title('Filtered image'), plt.xticks([]), plt.yticks([])
    
    plt.show()
    