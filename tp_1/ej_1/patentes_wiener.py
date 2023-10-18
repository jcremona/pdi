import cv2
import numpy as np
from matplotlib import pyplot as plt

# Source code from OpenCV samples
def motion_kernel(angle, d, sz):
    kern = np.ones((1, d), np.float32)

    # Build affine matrix A
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((d-1)*0.5, 0))

    # Apply affine transformation to kern
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
    return kern

def wiener_filter(image, angle_rad, d, nsr, psf_size=30):
    # Get psf
    psf = motion_kernel(angle_rad, d, sz=psf_size)
    
    # Apply padding to psf
    psf_pad = np.zeros_like(image)    
    kernel_h = psf.shape[0]
    kernel_w = psf.shape[1]
    psf_pad[:kernel_h, :kernel_w] = psf
    
    # Solution of applying Wiener filter in the frequency domain
    H = np.fft.fft2(psf_pad)
    H_spectrum = np.abs(H)
    H_sq_spectrum = np.square(H_spectrum)    
    G = np.fft.fft2(image)
    F_hat = G * H_sq_spectrum / (H * (H_sq_spectrum + nsr))
    f_hat = np.fft.ifft2(F_hat)
    img_filtered = np.real(f_hat)

    # Shift image
    img_filtered = np.roll(img_filtered, psf_size//2, 0)
    img_filtered = np.roll(img_filtered, psf_size//2, 1)

    # Normalize image
    img_filtered = cv2.normalize(img_filtered, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img_filtered

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='''
    Wiener filter
    ''')
    parser.add_argument('image_path', help='Path to image')
    parser.add_argument('--angle', type=int, help='Angle for motion kernel (in degrees)')
    parser.add_argument('--d', type=int, help='Distance for motion kernel (in degrees)')
    parser.add_argument('--nsr', type=float, help='Noise-signal ration (approximation with a constant)')
    args = parser.parse_args()

    image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    angle_deg = args.angle
    d = args.d
    nsr = args.nsr
    
    img_filtered = wiener_filter(image, np.deg2rad(angle_deg), d, nsr)

    # Display the image
    cv2.imshow('Original', image)
    cv2.imshow('Fixed', img_filtered)
            
    # Wait for a key press and then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
