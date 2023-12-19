import os
import cv2
import argparse

def match_images(image_list, min_th, path_to_images):
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Initialize the list of key images with the first image
    key_images = [image_list[0]]

    # Read the first image as the initial key image
    key_img_path = os.path.join(path_to_images, key_images[-1])
    key_img = cv2.imread(key_img_path, cv2.IMREAD_GRAYSCALE)
    keypoints1, descriptors1 = orb.detectAndCompute(key_img, None)

    for img_name in image_list[1:]:
        # Load next image
        next_img_path = os.path.join(path_to_images, img_name)
        next_img = cv2.imread(next_img_path, cv2.IMREAD_GRAYSCALE)
        keypoints2, descriptors2 = orb.detectAndCompute(next_img, None)

        # Match descriptors
        matches = bf.match(descriptors1, descriptors2)

        # Check if the matched points are below the threshold
        if len(matches) < min_th:
            # Update key image
            key_images.append(img_name)
            descriptors1 = descriptors2

    # Ensure the last image is a key image
    if image_list[-1] not in key_images:
        key_images.append(image_list[-1])

    return key_images

def save_key_images(key_images, path_to_images, output_folder):
    output_color = output_folder + "color/"
    output_depth = output_folder + "depth/"

    if not os.path.exists(output_color):
        os.makedirs(output_color)
    if not os.path.exists(output_depth):
        os.makedirs(output_depth)

    path_to_color = path_to_images + "color/"
    path_to_depth = path_to_images + "depth/"

    for img_name in key_images:
        img_path = os.path.join(path_to_color, img_name)
        img = cv2.imread(img_path)
        cv2.imwrite(os.path.join(output_color, img_name), img)
    
    for img_name in key_images:
        img_path = os.path.join(path_to_depth, img_name)
        img = cv2.imread(img_path)
        cv2.imwrite(os.path.join(output_depth, img_name), img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Key Image Selector")
    parser.add_argument("--input-folder", required=True, help="Path to the input folder containing the images.")
    parser.add_argument("--threshold", type=int, default=6, help="Minimum number of matched points to consider an image as a key image.")
    parser.add_argument("--output-folder", required=True, help="Path to the output folder where key images will be saved.")

    args = parser.parse_args()

    # Sort the images by name (or any other criterion you deem necessary)
    image_list = sorted([img for img in os.listdir(args.input_folder+"color/") if img.endswith('.png')])

    # Match images and get the list of key images
    key_images = match_images(image_list, args.threshold, args.input_folder+"color/")

    # Save the key images to the specified output folder
    save_key_images(key_images, args.input_folder, args.output_folder)
