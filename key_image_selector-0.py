import cv2
import torch
import numpy as np
from models.matching import Matching
from models.utils import frame2tensor
import shutil

def save_selected_images(visual_memory, src_color_path, src_depth_path, dst_color_path, dst_depth_path):
    for index in visual_memory:
        color_image_name = f'{index:04}.png'
        depth_image_name = f'{index:04}.png'  # Change this if depth image format is different

        src_color_image_path = src_color_path + color_image_name
        src_depth_image_path = src_depth_path + depth_image_name

        dst_color_image_path = dst_color_path + color_image_name
        dst_depth_image_path = dst_depth_path + depth_image_name

        shutil.copy(src_color_image_path, dst_color_image_path)
        shutil.copy(src_depth_image_path, dst_depth_image_path)

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error loading image from {image_path}")
    return image

def process_and_match_images(image1, image2, device):
    frame_tensor1 = frame2tensor(image1, device)
    frame_tensor2 = frame2tensor(image2, device)
    
    matching = Matching({'superpoint': {}, 'superglue': {'weights': 'indoor'}}).to(device).eval()
    with torch.no_grad():  # No need to track gradients
        pred = matching({'image0': frame_tensor1, 'image1': frame_tensor2})
    
    # Detach tensors before converting to NumPy arrays
    kpts0 = pred['keypoints0'][0].cpu().detach().numpy()
    kpts1 = pred['keypoints1'][0].cpu().detach().numpy()
    matches = pred['matches0'][0].cpu().detach().numpy()
    confidence = pred['matching_scores0'][0].cpu().detach().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    conf = confidence[valid]

    return mkpts0, mkpts1, conf

def draw_keypoints(image, keypoints):
    for keypoint in keypoints:
        x, y = map(int, keypoint)
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
    return image

def draw_matches(image1, image2, mkpts0, mkpts1, scale=1):
    """ Draw lines connecting matched keypoints between image1 and image2. """
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    height = max(h1, h2)
    width = w1 + w2
    matched_image = np.zeros((height, width, 3), dtype=np.uint8)
    matched_image[:h1, :w1] = image1
    matched_image[:h2, w1:w1+w2] = image2

    for pt1, pt2 in zip(mkpts0, mkpts1):
        pt1 = (int(round(pt1[0])), int(round(pt1[1])))
        pt2 = (int(round(pt2[0] + w1)), int(round(pt2[1])))

        cv2.circle(matched_image, pt1, 3, (0, 255, 0), -1)
        cv2.circle(matched_image, pt2, 3, (0, 255, 0), -1)
        cv2.line(matched_image, pt1,pt2, (255, 0, 0), 1)

    # Resize for visualization
    matched_image = cv2.resize(matched_image, (width // scale, height // scale))

    return matched_image

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")
    th_conf = 0.8
    th_num_matches = 100
    trusted_matches_list = []

    rootpath = "/home/rodriguez/Documents/logs/all_images/color/"
    num_images = 500  # Set this to the number of images in the folder
    visual_memory = [0]  # Initialize visual memory list with the first image index

    current_key_image = load_image(rootpath + '0000.png')

    for i in range(1, num_images):
        target_image_path = rootpath + f'{i:04}.png'
        target_image = load_image(target_image_path)

        mkpts0, mkpts1, confidences = process_and_match_images(current_key_image, target_image, device)

        # Filter matches by confidence
        trusted_matches = sum(conf > th_conf for conf in confidences)
        trusted_matches_list.append(trusted_matches)

        if trusted_matches < th_num_matches or i == num_images - 1:
            visual_memory.append(i)
            current_key_image = target_image

    print("Trusted Number of Matched Points: ", trusted_matches_list)
    print("Visual Memory Size: ", len(visual_memory))
    print("Visual Memory List:", visual_memory)

    src_color_path = "/home/rodriguez/Documents/logs/all_images/color/"
    src_depth_path = "/home/rodriguez/Documents/logs/all_images/depth/"
    dst_color_path = "/home/rodriguez/Documents/logs/vm/color/"
    dst_depth_path = "/home/rodriguez/Documents/logs/vm/depth/"

    save_selected_images(visual_memory, src_color_path, src_depth_path, dst_color_path, dst_depth_path)


if __name__ == "__main__":
    main()