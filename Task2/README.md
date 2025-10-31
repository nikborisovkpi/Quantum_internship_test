# Task 2. Computer vision. Sentinel-2 image matching
To perform the image matching task, I used a dataset from Kaggle (https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine/data) and downloaded it to the 'D:/deforestation-in-ukraine/' folder on a personal computer. The dataset contains satellite images taken at different times of the year.

### Repository structure:
1) Dataset_notebook.ipynb - notebook that explains the process of the dataset creation. The main parts of notebook:
   - The ImageLoader - universal image loader class and filter with the ability to save data.
   I created a template, self.pattern = '**/T36UYA_*_TCI.jp2', so the class loads only the required images. Photos with this template represent identical areas of land but were taken at different times of the year; such files are ideal for image matching.
   I also implemented logic for checking image quality, as the selected dataset contained many sampled photos with insufficient information.
   - load_images - function that convenient wrapper for bulk image loading with the ability to limit the number. Usee this function to load image dataset to slected folder(save_dataset_path = './dev_dataset/').
   - After that, we check some parameters of the dataset to check its quality and output its entire image via plt.subplots().

2) algorithm.py - python script (.py) for algorithm creation. The file contains one FeatureMatcher class - a class for finding, comparing and visualizing matches between two images using SIFT and Brute-Force Matcher. Key functions for work:
   - find_features_px - takes two images (numpy arrays), normalizes them, converts them to grayscale, finds keypoints and descriptors for both images via SIFT, returns the processed images, keypoints and descriptors.
   - find_features_path - takes paths to two images, loads the images from disk, checks that the images loaded successfully, calls find_features_px to find keypoints and descriptors.
   - compare_features - compares descriptors of two images using Brute-Force Matcher (knnMatch), applies ratio test: keeps only good matches (distance < threshold * n.distance), prints the number of good matches, returns a list of good matches.
   - draw_matches - function accepts image paths, finds keypoints and descriptors via find_features_path, compares descriptors via compare_features, draws matches between images using cv2.drawMatches, and returns an image with the drawn matches.

3) Inference.py - file with main() function, which displays the img_path_1 and img_path_2 images, as well as their matches using the draw_matches() function from the algorithm.py file. The function highlights good matches (distance < threshold * n.distance).

4) Demo_notebook.ipynb - Jupyter notebook with demo. Uses all functions and methods from algorithm.py and Inference.py to display images and their matches on the screen.



### Usage example:
First, let's navigate to the Task2 folder:
```
cd Task2
```

Now you need to install all packages from requirements file:
```
pip install -r requirements.txt
```

To use FeatureMatcher class, display images and their matches on the screendisplay:
```
python Inference.py --img_path_1 T36UYA_20190825T083601_TCI --img_path_2 T36UYA_20190706T083611_TCI --threshold 0.65
```