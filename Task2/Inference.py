import cv2
import matplotlib.pyplot as plt
import argparse

from algorithm import FeatureMatcher


def main(args):
    f_matcher = FeatureMatcher()
    img_path_1 = './dev_dataset/' + args.img_path_1 + '.jpg'
    img_path_2 = './dev_dataset/' + args.img_path_2 + '.jpg'

    img1 = cv2.imread(img_path_1)
    img2 = cv2.imread(img_path_2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Display input images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img1)
    ax1.set_title('Image 1')
    ax1.axis('off')

    ax2.imshow(img2)
    ax2.set_title('Image 2')
    ax2.axis('off')
    plt.show()

    # Draw matched features
    matched_images = f_matcher.draw_matches(img_path_1, img_path_2, args.threshold)
    plt.figure(figsize=(10, 5))
    plt.imshow(matched_images)
    plt.title('Matched Features')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path_1', type=str, required=True, help='Path to first image')
    parser.add_argument('--img_path_2', type=str, required=True, help='Path to second image')
    parser.add_argument('--threshold', type=float, default=0.75, help='Threshold for feature matching')
    args = parser.parse_args()
    main(args)