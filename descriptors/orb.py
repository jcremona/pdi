import cv2


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='''
    ORB
    ''')
    parser.add_argument('image_path', help='Path to image')
    args = parser.parse_args()

    # Read an image
    image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)

    # Draw keypoints on the image
    output_image = cv2.drawKeypoints(image, keypoints, None)

    # Display the result
    cv2.imshow('ORB Keypoints', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
