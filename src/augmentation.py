"""
This file contains a collection of functions and image processing
techniques to augment the input images to train the DETR models on
the task of Table Detection and Table structure recognition.


Author:
    Name:
        Swathi Thiruvengadam
    Email:
        swathi.thiruvengadam@ise.fraunhofer.de
        swathi.thiru078@gmail.com
"""

import os
import numpy as np
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image

def augmentImages(input_folder, output_folder):
    """
    this function applies random partial masking to images in the input folder
    and save augmented images to the output folder.

    Parameters:
    - input_folder (str): Path to the folder containing input images.
    - output_folder (str): Path to the folder where augmented images will be saved.

    Returns:
    - None: The function saves the augmented images in the output folder.
    """

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_folder, filename)
            img = load_img(img_path)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            # Image with Random Partial Mask
            min_mask_size = (int(x.shape[1] * 0.4), int(x.shape[2] * 0.4))

            # Randomly determine the mask size within the calculated minimum mask size range
            mask_size = (
                np.random.randint(min_mask_size[0], int(x.shape[1] * 0.8)),
                np.random.randint(min_mask_size[1], int(x.shape[2] * 0.8))
            )

            # Randomly determine the mask position
            mask_position = (
                np.random.randint(0, x.shape[1] - mask_size[0]),
                np.random.randint(0, x.shape[2] - mask_size[1])
            )

            mask_color = np.random.randint(0, 256, size=(3,))  # Random color
            mask_opacity = 0.2  # Adjust the opacity as needed

            # Create the masked image
            masked_image = x.copy()
            masked_image[:, mask_position[0]:mask_position[0] + mask_size[0],
            mask_position[1]:mask_position[1] + mask_size[1], :] = (
                                                                           1 - mask_opacity
                                                                   ) * masked_image[:,
                                                                       mask_position[0]:mask_position[0] + mask_size[0],
                                                                       mask_position[1]:mask_position[1] + mask_size[1],
                                                                       :] + (
                                                                       mask_opacity
                                                                   ) * mask_color.reshape((1, 1, 3))

            masked_image = array_to_img(masked_image[0])  # Convert back to PIL image
            output_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_mask.jpg')
            masked_image.save(output_path)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', required=True, help="Directory for input images")
    parser.add_argument('--out_dir', required=True, help="Directory for output images")
    return parser.parse_args()

def main():
    args = get_args()
    print(args.__dict__)

    augmentImages(args.in_dir, args.out_dir)

if __name__ == "__main__":
    main()

