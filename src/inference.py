from collections import OrderedDict, defaultdict
import json
import argparse
import sys
import xml.etree.ElementTree as ET
import os
import yaml
import openai
from os import listdir
from os.path import join, split
#from spellchecker import SpellChecker
import random
from scipy import stats
import csv
from io import StringIO

import math
import time
import cv2


import torch
import pytesseract
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
from fitz import Rect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
from pdf2image import convert_from_path
from finalstep import Datasheet

from main import get_model
import postprocess
#sys.path.append('../detr/models')
sys.path.append('../')
from detr.models import build_model

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))

        return resized_image

detection_transform = transforms.Compose([
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

structure_transform = transforms.Compose([
    MaxResize(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_class_map(data_type):
    if data_type == 'structure':
        class_map = {
            'table': 0,
            'table column': 1,
            'table row': 2,
            'table column header': 3,
            'table projected row header': 4,
            'table spanning cell': 5,
            'no object': 6,
            'table row header': 7,
            'table projected column header': 8,
            'table name': 9
        }
    elif data_type == 'detection':
        class_map = {'table': 0, 'table rotated': 1, 'no object': 2}
    return class_map

detection_class_thresholds = {
    "table": 0.8,
    "table rotated": 0.7,
    "no object": 10
}

structure_class_thresholds = {
    "table": 0.8,
    "table column": 0.6,
    "table row": 0.6,
    "table column header": 0.6,
    "table projected row header": 0.3,
    "table spanning cell": 0.4,
    "no object": 10,
    "table row header": 0.6,
    "table projected column header": 0.3,
    "table name": 0.6
}

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_dir',
                        help="Directory for input images")
    parser.add_argument('--words_dir',
                        help="Directory for input words")
    parser.add_argument('--out_dir',
                        help="Output directory")
    parser.add_argument('--model_dir',
                        help="Directory containing the models")
    parser.add_argument('--mode',
                        help="The processing to apply to the input image and tokens",
                        choices=['detect', 'recognize', 'extract'])
    parser.add_argument('--structure_config_path',
                        help="Filepath to the structure model config file")
    parser.add_argument('--structure_model_path', help="The path to the structure model")
    parser.add_argument('--detection_config_path',
                        help="Filepath to the detection model config file")
    parser.add_argument('--detection_model_path', help="The path to the detection model")
    parser.add_argument('--detection_device', default="cuda")
    parser.add_argument('--structure_device', default="cuda")
    parser.add_argument('--crops', '-p', action='store_true',
                        help='Output cropped data from table detections')
    parser.add_argument('--objects', '-o', action='store_true',
                        help='Output objects')
    parser.add_argument('--cells', '-l', action='store_true',
                        help='Output cells list')
    parser.add_argument('--html', '-m', action='store_true',
                        help='Output HTML')
    parser.add_argument('--csv', '-c', action='store_true',
                        help='Output CSV')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--visualize', '-z', action='store_true',
                        help='Visualize output')
    parser.add_argument('--crop_padding', type=int, default=10,
                        help="The amount of padding to add around a detected table when cropping.")

    return parser.parse_args()


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    output = torch.stack(b, dim=1)
    return output


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def iob(bbox1, bbox2):
    """
    Compute the intersection area over box area, for bbox1.
    """
    intersection = Rect(bbox1).intersect(bbox2)

    bbox1_area = Rect(bbox1).get_area()
    if bbox1_area > 0:
        iob_ratio = intersection.get_area() / bbox1_area
        return iob_ratio

    return 0


def enhance_image(image):
    """This function enhances a given image by applying several preprocessing steps,
    including converting to grayscale, reducing noise with Gaussian blur, binarization,
    and sharpening. It returns the processed image as a PIL Image object.

    Parameters:
        image: a PIL Image object representing the input image to be enhanced.

    Returns:
        final_img: a PIL Image object representing the enhanced, sharpened, and binarized
                   version of the input image.

    Example:
        #>>> img = Image.open("sample_image.jpg")
        #>>> enhanced_img = enhance_image(img)
        #>>> enhanced_img.show()  # Opens the enhanced image for display
    """

    # Convert PIL image to OpenCV format
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # 1. Convert to Grayscale
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 2. Apply GaussianBlur to reduce noise
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # 3. Apply Thresholding (Binarization)
    _, binary_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Sharpen the Image using kernel filter
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_img = cv2.filter2D(binary_img, -1, kernel)

    # 5. Optional: Rescale the image (if too small, you can upscale it)
    #resized_img = cv2.resize(sharpened_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    # 6. Deskew the image (optional)
    # You can add deskewing if you know the image is skewed

    # Convert back to PIL Image for saving
    final_img = Image.fromarray(sharpened_img)

    return final_img

def align_headers(headers, rows):
    """
    Adjust the header boundary to be the convex hull of the rows it intersects
    at least 50% of the height of.

    For now, we are not supporting tables with multiple headers, so we need to
    eliminate anything besides the top-most header.
    """

    aligned_headers = []

    for row in rows:
        row['column header'] = False

    header_row_nums = []
    for header in headers:
        for row_num, row in enumerate(rows):
            row_height = row['bbox'][3] - row['bbox'][1]
            min_row_overlap = max(row['bbox'][1], header['bbox'][1])
            max_row_overlap = min(row['bbox'][3], header['bbox'][3])
            overlap_height = max_row_overlap - min_row_overlap
            if overlap_height / row_height >= 0.5:
                header_row_nums.append(row_num)

    if len(header_row_nums) == 0:
        return aligned_headers

    header_rect = Rect()
    if header_row_nums[0] > 0:
        header_row_nums = list(range(header_row_nums[0] + 1)) + header_row_nums

    last_row_num = -1
    for row_num in header_row_nums:
        if row_num == last_row_num + 1:
            row = rows[row_num]
            row['column header'] = True
            header_rect = header_rect.include_rect(row['bbox'])
            last_row_num = row_num
        else:
            # Break as soon as a non-header row is encountered.
            # This ignores any subsequent rows in the table labeled as a header.
            # Having more than 1 header is not supported currently.
            break

    header = {'bbox': list(header_rect)}
    aligned_headers.append(header)

    return aligned_headers


def refine_table_structure(table_structure, class_thresholds):
    """
    Apply operations to the detected table structure objects such as
    thresholding, NMS, and alignment.
    """
    rows = table_structure["rows"]
    columns = table_structure['columns']
    table_name = table_structure['table name']

    # Process the headers
    column_headers = table_structure['column headers']
    column_headers = postprocess.apply_threshold(column_headers, class_thresholds["table column header"])
    column_headers = postprocess.nms(column_headers)
    column_headers = align_headers(column_headers, rows)

    # Process the row headers
    row_headers = table_structure['row headers']
    row_headers = postprocess.apply_threshold(row_headers, class_thresholds["table row header"])
    row_headers = postprocess.nms(row_headers)
    row_headers = align_headers(row_headers, columns)

    '''# Process spanning cells  ~~ original
    spanning_cells = [elem for elem in table_structure['spanning cells'] if not elem['projected row header']]
    projected_row_headers = [elem for elem in table_structure['spanning cells'] if elem['projected row header']]
    spanning_cells = postprocess.apply_threshold(spanning_cells, class_thresholds["table spanning cell"])
    projected_row_headers = postprocess.apply_threshold(projected_row_headers,
                                                        class_thresholds["table projected row header"])
    spanning_cells += projected_row_headers'''

    # Process spanning cells
    spanning_cells = [elem for elem in table_structure['spanning cells'] if not elem['projected row header'] and not elem['projected column header']]
    projected_row_headers = [elem for elem in table_structure['spanning cells'] if elem['projected row header']]

    projected_column_headers = [elem for elem in table_structure['spanning cells'] if elem['projected column header']]
    spanning_cells = postprocess.apply_threshold(spanning_cells, class_thresholds["table spanning cell"])
    projected_row_headers = postprocess.apply_threshold(projected_row_headers,
                                                        class_thresholds["table projected row header"])
    projected_column_headers = postprocess.apply_threshold(projected_column_headers,
                                                        class_thresholds["table projected column header"])
    spanning_cells += projected_row_headers
    # Align before NMS for spanning cells because alignment brings them into agreement
    # with rows and columns first; if spanning cells still overlap after this operation,
    # the threshold for NMS can basically be lowered to just above 0
    spanning_cells = postprocess.align_supercells(spanning_cells, rows, columns)
    spanning_cells = postprocess.nms_supercells(spanning_cells)

    postprocess.header_supercell_tree(spanning_cells)

    table_structure['table_name'] = table_name
    table_structure['columns'] = columns
    table_structure['rows'] = rows
    table_structure['spanning cells'] = spanning_cells
    table_structure['column headers'] = column_headers
    table_structure['row headers'] = row_headers

    return table_structure

def extract_words_from_images(input_folder, output_folder):
    '''This function takes 2 paths as input:
        for all image file in the input_folder_path it will extract the words
        using an OCR-Pytesserct and saves the extracted file in the Output_folder_path
        Parameters:
                input_folder:
        '''

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    # Identify all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', 'PNG', '.jpg', '.jpeg', '.gif', '.bmp'))]

    for image_file in image_files:
        try:
            # Finding path to input images
            image_path = os.path.join(input_folder, image_file)
            with Image.open(image_path).convert('RGB') as img:
                # improve the input image
                img = img.filter(ImageFilter.SHARPEN)
                img = enhance_image(img)
                # Use Tesseract to do OCR on the image
                extracted_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                #spell = SpellChecker()

                words_data = [
                    {"bbox": [extracted_data['left'][i], extracted_data['top'][i],
                              extracted_data['left'][i] + extracted_data['width'][i],
                              extracted_data['top'][i] + extracted_data['height'][i]],
                     "text": extracted_data['text'][i].strip(),
                     "line_num": extracted_data['line_num'][i],
                     "block_num": extracted_data['block_num'][i]}
                    for i in range(len(extracted_data['text'])) if extracted_data['text'][i].strip()
                ]

                # Construct the full path to the output JSON file (same name as image file with .json extension)
                output_json_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + "_words.json")

                # Write the extracted word data to a JSON file
                with open(output_json_path, 'w') as json_file:
                    json.dump(words_data, json_file)
                    print("OCR file created")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

def outputs_to_objects(outputs, img_size, class_idx2name):
    m = outputs['pred_logits'].softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = class_idx2name[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects


def objects_to_crops(img, tokens, objects, class_thresholds, padding=20):
    """This function extracts cropped image regions and their tokens)
    based on detected objects from the image, and adjusts the bounding boxes of
    tokens. Optionally adds padding to the crops and handles rotation
    for specific object labels (e.g., rotated tables).

    Parameters:
        img: the input image (PIL.Image) from which crops will be taken.
        tokens: a list of token dictionaries, each containing 'bbox' (bounding box
                coordinates) and other token information.
        objects: a list of dictionaries where each represents a detected object with
                 keys 'label', 'score', and 'bbox'.
        class_thresholds: a dictionary mapping object labels to the minimum confidence
                          score required for an object to be considered.
        padding: an integer specifying how many pixels of padding to add around each
                 object's bounding box (default is 20).

    Returns:
        table_crops: a list of dictionaries, each containing the cropped image ('image')
                     and the adjusted tokens ('tokens') found within the cropped region.

    Example:
        >>> img = Image.open("../images/Eging_2.jpg")
        >>> tokens = [{'bbox': [165, 277, 1060, 360], 'text': 'EG-455M72-HE/BF-DG', 'line_num': 1, 'block_num': 1, 'span_num': 0}, {'bbox': [177, 512, 418, 557], 'text': 'Engineering', 'line_num': 1, 'block_num': 3, 'span_num': 1}, {'bbox': [432, 512, 617, 557], 'text': 'Drawings', 'line_num': 1, 'block_num': 3, 'span_num': 2}, {'bbox': [573, 1004, 655, 1032], 'text': '—_-—-—_-', 'line_num': 1, 'block_num': 16, 'span_num': 3}, {'bbox': [778, 1014, 820, 1027], 'text': '=', 'line_num': 1, 'block_num': 16, 'span_num': 4}, {'bbox': [899, 1014, 1016, 1026], 'text': '=—-o', 'line_num': 1, 'block_num': 16, 'span_num': 5}, {'bbox': [989, 1086, 1057, 1098], 'text': 'j+__}', 'line_num': 2, 'block_num': 16, 'span_num': 6}, {'bbox': [373, 1874, 399, 1908], 'text': 'S', 'line_num': 1, 'block_num': 16, 'span_num': 7}, {'bbox': [377, 1915, 397, 1940], 'text': '€', 'line_num': 1, 'block_num': 16, 'span_num': 8}, {'bbox': [381, 1943, 397, 1968], 'text': '2', 'line_num': 1, 'block_num': 16, 'span_num': 9}, {'bbox': [381, 1970, 397, 1997], 'text': '5', 'line_num': 1, 'block_num': 16, 'span_num': 10}, {'bbox': [376, 2000, 398, 2016], 'text': 'oO', 'line_num': 1, 'block_num': 16, 'span_num': 11}, {'bbox': [531, 2130, 617, 2145], 'text': '(as', 'line_num': 1, 'block_num': 16, 'span_num': 12}, {'bbox': [688, 2143, 692, 2145], 'text': 'F', 'line_num': 1, 'block_num': 16, 'span_num': 13}, {'bbox': [600, 2164, 700, 2194], 'text': 'Voltage', 'line_num': 2, 'block_num': 16, 'span_num': 14}, {'bbox': [707, 2163, 740, 2189], 'text': '(V)', 'line_num': 2, 'block_num': 16, 'span_num': 15}, {'bbox': [374, 2328, 400, 2362], 'text': 'S', 'line_num': 1, 'block_num': 16, 'span_num': 16}, {'bbox': [378, 2369, 398, 2394], 'text': '€', 'line_num': 1, 'block_num': 16, 'span_num': 17}, {'bbox': [382, 2397, 398, 2421], 'text': '“', 'line_num': 1, 'block_num': 16, 'span_num': 18}, {'bbox': [382, 2424, 398, 2451], 'text': '5', 'line_num': 1, 'block_num': 16, 'span_num': 19}, {'bbox': [377, 2454, 399, 2470], 'text': 'o', 'line_num': 1, 'block_num': 16, 'span_num': 20}, {'bbox': [612, 2618, 712, 2648], 'text': 'Voltage', 'line_num': 1, 'block_num': 16, 'span_num': 21}, {'bbox': [719, 2617, 752, 2643], 'text': '(V)', 'line_num': 1, 'block_num': 16, 'span_num': 22}, {'bbox': [178, 2730, 334, 2775], 'text': 'Packing', 'line_num': 2, 'block_num': 16, 'span_num': 23}, {'bbox': [347, 2729, 623, 2775], 'text': 'Configuration', 'line_num': 2, 'block_num': 16, 'span_num': 24}, {'bbox': [177, 2793, 250, 2813], 'text': 'Pieces', 'line_num': 3, 'block_num': 16, 'span_num': 25}, {'bbox': [259, 2799, 297, 2819], 'text': 'per', 'line_num': 3, 'block_num': 16, 'span_num': 26}, {'bbox': [304, 2793, 370, 2819], 'text': 'pallet', 'line_num': 3, 'block_num': 16, 'span_num': 27}, {'bbox': [649, 2797, 676, 2816], 'text': '36', 'line_num': 3, 'block_num': 16, 'span_num': 28}, {'bbox': [177, 2844, 223, 2864], 'text': 'Size', 'line_num': 4, 'block_num': 16, 'span_num': 29}, {'bbox': [230, 2844, 254, 2864], 'text': 'of', 'line_num': 4, 'block_num': 16, 'span_num': 30}, {'bbox': [261, 2844, 353, 2870], 'text': 'packing', 'line_num': 4, 'block_num': 16, 'span_num': 31}, {'bbox': [364, 2843, 425, 2866], 'text': '(mm)', 'line_num': 4, 'block_num': 16, 'span_num': 32}, {'bbox': [648, 2843, 841, 2862], 'text': '2130°1140°1190', 'line_num': 4, 'block_num': 16, 'span_num': 33}, {'bbox': [176, 2894, 258, 2922], 'text': 'Weight', 'line_num': 5, 'block_num': 16, 'span_num': 34}, {'bbox': [265, 2894, 289, 2914], 'text': 'of', 'line_num': 5, 'block_num': 16, 'span_num': 35}, {'bbox': [296, 2894, 388, 2922], 'text': 'packing', 'line_num': 5, 'block_num': 16, 'span_num': 36}, {'bbox': [399, 2893, 442, 2922], 'text': '(kg)', 'line_num': 5, 'block_num': 16, 'span_num': 37}, {'bbox': [650, 2890, 704, 2909], 'text': '1040', 'line_num': 5, 'block_num': 16, 'span_num': 38}, {'bbox': [177, 2943, 196, 2963], 'text': 'Pi', 'line_num': 6, 'block_num': 16, 'span_num': 39}, {'bbox': [289, 2943, 416, 2963], 'text': 'rcontainer', 'line_num': 6, 'block_num': 16, 'span_num': 40}, {'bbox': [650, 2939, 691, 2958], 'text': '792', 'line_num': 6, 'block_num': 16, 'span_num': 41}, {'bbox': [647, 2985, 717, 3004], 'text': '40’HC', 'line_num': 7, 'block_num': 16, 'span_num': 42}, {'bbox': [177, 2992, 223, 3012], 'text': 'Size', 'line_num': 1, 'block_num': 17, 'span_num': 43}, {'bbox': [230, 2992, 254, 3012], 'text': 'of', 'line_num': 1, 'block_num': 17, 'span_num': 44}, {'bbox': [260, 2992, 373, 3012], 'text': 'container', 'line_num': 1, 'block_num': 17, 'span_num': 45}, {'bbox': [152, 3173, 248, 3194], 'text': 'Revised', 'line_num': 1, 'block_num': 19, 'span_num': 46}, {'bbox': [257, 3173, 278, 3194], 'text': 'in', 'line_num': 1, 'block_num': 19, 'span_num': 47}, {'bbox': [287, 3175, 338, 3194], 'text': 'MAY', 'line_num': 1, 'block_num': 19, 'span_num': 48}, {'bbox': [345, 3175, 404, 3194], 'text': '2022', 'line_num': 1, 'block_num': 19, 'span_num': 49}, {'bbox': [413, 3174, 452, 3194], 'text': '1th', 'line_num': 1, 'block_num': 19, 'span_num': 50}, {'bbox': [461, 3173, 549, 3194], 'text': 'Edition', 'line_num': 1, 'block_num': 19, 'span_num': 51}, {'bbox': [150, 3214, 296, 3234], 'text': 'CAUTION:AII', 'line_num': 1, 'block_num': 20, 'span_num': 52}, {'bbox': [305, 3214, 371, 3240], 'text': 'rights', 'line_num': 1, 'block_num': 20, 'span_num': 53}, {'bbox': [379, 3214, 480, 3234], 'text': 'reserved', 'line_num': 1, 'block_num': 20, 'span_num': 54}, {'bbox': [490, 3214, 516, 3240], 'text': 'by', 'line_num': 1, 'block_num': 20, 'span_num': 55}, {'bbox': [524, 3215, 597, 3234], 'text': 'EGING', 'line_num': 1, 'block_num': 20, 'span_num': 56}, {'bbox': [606, 3215, 639, 3234], 'text': 'PV.', 'line_num': 1, 'block_num': 20, 'span_num': 57}, {'bbox': [151, 3249, 317, 3275], 'text': 'Specifications', 'line_num': 2, 'block_num': 20, 'span_num': 58}, {'bbox': [325, 3249, 426, 3269], 'text': 'included', 'line_num': 2, 'block_num': 20, 'span_num': 59}, {'bbox': [435, 3249, 454, 3269], 'text': 'in', 'line_num': 2, 'block_num': 20, 'span_num': 60}, {'bbox': [463, 3249, 505, 3269], 'text': 'this', 'line_num': 2, 'block_num': 20, 'span_num': 61}, {'bbox': [513, 3249, 630, 3269], 'text': 'datasheet', 'line_num': 2, 'block_num': 20, 'span_num': 62}, {'bbox': [637, 3255, 673, 3269], 'text': 'are', 'line_num': 2, 'block_num': 20, 'span_num': 63}, {'bbox': [681, 3249, 766, 3275], 'text': 'subject', 'line_num': 2, 'block_num': 20, 'span_num': 64}, {'bbox': [774, 3252, 796, 3269], 'text': 'to', 'line_num': 2, 'block_num': 20, 'span_num': 65}, {'bbox': [804, 3249, 889, 3275], 'text': 'change', 'line_num': 2, 'block_num': 20, 'span_num': 66}, {'bbox': [897, 3249, 989, 3269], 'text': 'without', 'line_num': 2, 'block_num': 20, 'span_num': 67}, {'bbox': [997, 3249, 1075, 3269], 'text': 'notice.', 'line_num': 2, 'block_num': 20, 'span_num': 68}, {'bbox': [1796, 158, 1916, 247], 'text': '&', 'line_num': 1, 'block_num': 21, 'span_num': 69}, {'bbox': [1971, 193, 2304, 238], 'text': 'EcIncpv', 'line_num': 1, 'block_num': 21, 'span_num': 70}, {'bbox': [1770, 259, 1948, 282], 'text': 'KEENSTAR', 'line_num': 1, 'block_num': 22, 'span_num': 71}, {'bbox': [1263, 512, 1447, 547], 'text': 'Electrical', 'line_num': 1, 'block_num': 23, 'span_num': 72}, {'bbox': [1461, 512, 1760, 547], 'text': 'Characteristics', 'line_num': 1, 'block_num': 23, 'span_num': 73}, {'bbox': [1265, 575, 1335, 594], 'text': 'Power', 'line_num': 1, 'block_num': 32, 'span_num': 74}, {'bbox': [1343, 574, 1393, 594], 'text': 'level', 'line_num': 1, 'block_num': 32, 'span_num': 75}, {'bbox': [1733, 575, 1775, 594], 'text': '435', 'line_num': 1, 'block_num': 32, 'span_num': 76}, {'bbox': [1851, 570, 1897, 605], 'text': '440', 'line_num': 1, 'block_num': 32, 'span_num': 77}, {'bbox': [1973, 570, 2018, 605], 'text': '445', 'line_num': 1, 'block_num': 32, 'span_num': 78}, {'bbox': [2099, 575, 2140, 594], 'text': '450', 'line_num': 1, 'block_num': 32, 'span_num': 79}, {'bbox': [2221, 575, 2262, 594], 'text': '455', 'line_num': 1, 'block_num': 32, 'span_num': 80}, {'bbox': [1265, 627, 1327, 646], 'text': 'Pmax', 'line_num': 2, 'block_num': 32, 'span_num': 81}, {'bbox': [1333, 625, 1370, 648], 'text': '(W)', 'line_num': 2, 'block_num': 32, 'span_num': 82}, {'bbox': [1733, 627, 1774, 646], 'text': '435', 'line_num': 2, 'block_num': 32, 'span_num': 83}, {'bbox': [1851, 623, 1896, 656], 'text': '440', 'line_num': 2, 'block_num': 32, 'span_num': 84}, {'bbox': [1976, 627, 2018, 646], 'text': '445', 'line_num': 2, 'block_num': 32, 'span_num': 85}, {'bbox': [2098, 627, 2140, 646], 'text': '450', 'line_num': 2, 'block_num': 32, 'span_num': 86}, {'bbox': [2220, 627, 2262, 646], 'text': '455', 'line_num': 2, 'block_num': 32, 'span_num': 87}, {'bbox': [1263, 677, 1315, 701], 'text': 'vmp', 'line_num': 3, 'block_num': 32, 'span_num': 88}, {'bbox': [1321, 674, 1350, 697], 'text': '(V)', 'line_num': 3, 'block_num': 32, 'span_num': 89}, {'bbox': [1722, 674, 1785, 707], 'text': '41.04', 'line_num': 3, 'block_num': 32, 'span_num': 90}, {'bbox': [1844, 678, 1907, 697], 'text': '41.24', 'line_num': 3, 'block_num': 32, 'span_num': 91}, {'bbox': [1966, 678, 2029, 697], 'text': '41.44', 'line_num': 3, 'block_num': 32, 'span_num': 92}, {'bbox': [2088, 678, 2151, 697], 'text': '41.63', 'line_num': 3, 'block_num': 32, 'span_num': 93}, {'bbox': [2209, 678, 2272, 697], 'text': '41.82', 'line_num': 3, 'block_num': 32, 'span_num': 94}, {'bbox': [1265, 728, 1308, 752], 'text': 'Imp', 'line_num': 4, 'block_num': 32, 'span_num': 95}, {'bbox': [1314, 725, 1344, 748], 'text': '(A)', 'line_num': 4, 'block_num': 32, 'span_num': 96}, {'bbox': [1723, 728, 1784, 747], 'text': '10.60', 'line_num': 4, 'block_num': 32, 'span_num': 97}, {'bbox': [1845, 728, 1906, 747], 'text': '10.67', 'line_num': 4, 'block_num': 32, 'span_num': 98}, {'bbox': [1967, 728, 2028, 747], 'text': '10.74', 'line_num': 4, 'block_num': 32, 'span_num': 99}, {'bbox': [2089, 728, 2149, 747], 'text': '10.81', 'line_num': 4, 'block_num': 32, 'span_num': 100}, {'bbox': [2211, 728, 2272, 747], 'text': '10.88', 'line_num': 4, 'block_num': 32, 'span_num': 101}, {'bbox': [1263, 781, 1303, 799], 'text': 'Voc', 'line_num': 5, 'block_num': 32, 'span_num': 102}, {'bbox': [1310, 778, 1339, 801], 'text': '(V)', 'line_num': 5, 'block_num': 32, 'span_num': 103}, {'bbox': [1722, 781, 1785, 800], 'text': '49.25', 'line_num': 5, 'block_num': 32, 'span_num': 104}, {'bbox': [1844, 777, 1907, 810], 'text': '49.44', 'line_num': 5, 'block_num': 32, 'span_num': 105}, {'bbox': [1966, 781, 2029, 800], 'text': '49.65', 'line_num': 5, 'block_num': 32, 'span_num': 106}, {'bbox': [2088, 781, 2151, 800], 'text': '49.85', 'line_num': 5, 'block_num': 32, 'span_num': 107}, {'bbox': [2210, 781, 2272, 800], 'text': '50.06', 'line_num': 5, 'block_num': 32, 'span_num': 108}, {'bbox': [1266, 824, 1294, 860], 'text': 'Isc', 'line_num': 6, 'block_num': 32, 'span_num': 109}, {'bbox': [1301, 828, 1330, 851], 'text': '(A)', 'line_num': 6, 'block_num': 32, 'span_num': 110}, {'bbox': [1723, 831, 1783, 849], 'text': '11.11', 'line_num': 6, 'block_num': 32, 'span_num': 111}, {'bbox': [1845, 830, 1906, 849], 'text': '11.17', 'line_num': 6, 'block_num': 32, 'span_num': 112}, {'bbox': [1967, 830, 2028, 849], 'text': '11.24', 'line_num': 6, 'block_num': 32, 'span_num': 113}, {'bbox': [2089, 830, 2149, 849], 'text': '11.31', 'line_num': 6, 'block_num': 32, 'span_num': 114}, {'bbox': [2211, 830, 2272, 849], 'text': '11.38', 'line_num': 6, 'block_num': 32, 'span_num': 115}, {'bbox': [1266, 878, 1347, 898], 'text': 'Module', 'line_num': 7, 'block_num': 32, 'span_num': 116}, {'bbox': [1355, 878, 1462, 904], 'text': 'efficiency', 'line_num': 7, 'block_num': 32, 'span_num': 117}, {'bbox': [1467, 877, 1503, 900], 'text': '(%)', 'line_num': 7, 'block_num': 32, 'span_num': 118}, {'bbox': [1723, 878, 1784, 897], 'text': '20.01', 'line_num': 7, 'block_num': 32, 'span_num': 119}, {'bbox': [1845, 879, 1907, 898], 'text': '20.24', 'line_num': 7, 'block_num': 32, 'span_num': 120}, {'bbox': [1966, 879, 2028, 898], 'text': '20.47', 'line_num': 7, 'block_num': 32, 'span_num': 121}, {'bbox': [2088, 878, 2150, 897], 'text': '20.70', 'line_num': 7, 'block_num': 32, 'span_num': 122}, {'bbox': [2210, 879, 2272, 898], 'text': '20.93', 'line_num': 7, 'block_num': 32, 'span_num': 123}, {'bbox': [1265, 933, 1375, 953], 'text': 'Maximum', 'line_num': 8, 'block_num': 32, 'span_num': 124}, {'bbox': [1383, 936, 1460, 959], 'text': 'system', 'line_num': 8, 'block_num': 32, 'span_num': 125}, {'bbox': [1468, 933, 1549, 959], 'text': 'voltage', 'line_num': 8, 'block_num': 32, 'span_num': 126}, {'bbox': [1554, 932, 1582, 955], 'text': '(V)', 'line_num': 8, 'block_num': 32, 'span_num': 127}, {'bbox': [1972, 934, 2026, 953], 'text': '1500', 'line_num': 8, 'block_num': 32, 'span_num': 128}, {'bbox': [1265, 982, 1315, 1001], 'text': 'Fuse', 'line_num': 1, 'block_num': 32, 'span_num': 129}, {'bbox': [1324, 981, 1394, 1007], 'text': 'Rating', 'line_num': 1, 'block_num': 32, 'span_num': 130}, {'bbox': [1399, 980, 1427, 1003], 'text': '(A)', 'line_num': 1, 'block_num': 32, 'span_num': 131}, {'bbox': [1985, 982, 2011, 1001], 'text': '20', 'line_num': 1, 'block_num': 32, 'span_num': 132}, {'bbox': [1265, 1030, 1408, 1055], 'text': 'Temperature', 'line_num': 1, 'block_num': 32, 'span_num': 133}, {'bbox': [1415, 1029, 1532, 1049], 'text': 'coefficient', 'line_num': 1, 'block_num': 32, 'span_num': 134}, {'bbox': [1547, 1030, 1609, 1049], 'text': 'Pmax', 'line_num': 1, 'block_num': 32, 'span_num': 135}, {'bbox': [1615, 1028, 1677, 1051], 'text': '(%°C)', 'line_num': 1, 'block_num': 32, 'span_num': 136}, {'bbox': [1963, 1030, 2033, 1049], 'text': '-0.350', 'line_num': 1, 'block_num': 32, 'span_num': 137}, {'bbox': [1265, 1082, 1408, 1107], 'text': 'Temperature', 'line_num': 1, 'block_num': 32, 'span_num': 138}, {'bbox': [1415, 1081, 1532, 1101], 'text': 'coefficient', 'line_num': 1, 'block_num': 32, 'span_num': 139}, {'bbox': [1547, 1082, 1575, 1100], 'text': 'Isc', 'line_num': 1, 'block_num': 32, 'span_num': 140}, {'bbox': [1581, 1079, 1643, 1102], 'text': '(%°C)', 'line_num': 1, 'block_num': 32, 'span_num': 141}, {'bbox': [1974, 1081, 2021, 1100], 'text': '0.05', 'line_num': 1, 'block_num': 32, 'span_num': 142}, {'bbox': [1265, 1133, 1408, 1158], 'text': 'Temperature', 'line_num': 1, 'block_num': 32, 'span_num': 143}, {'bbox': [1415, 1132, 1532, 1152], 'text': 'coefficient', 'line_num': 1, 'block_num': 32, 'span_num': 144}, {'bbox': [1545, 1135, 1585, 1153], 'text': 'Voc', 'line_num': 1, 'block_num': 32, 'span_num': 145}, {'bbox': [1591, 1132, 1653, 1155], 'text': '(%°C)', 'line_num': 1, 'block_num': 32, 'span_num': 146}, {'bbox': [1963, 1135, 2033, 1154], 'text': '-0.275', 'line_num': 1, 'block_num': 32, 'span_num': 147}, {'bbox': [1266, 1181, 1436, 1201], 'text': 'STC:Irradiance', 'line_num': 1, 'block_num': 34, 'span_num': 148}, {'bbox': [1446, 1179, 1664, 1205], 'text': '1000W/m?,module', 'line_num': 1, 'block_num': 34, 'span_num': 149}, {'bbox': [1671, 1184, 1819, 1206], 'text': 'temperature', 'line_num': 1, 'block_num': 34, 'span_num': 150}, {'bbox': [1828, 1182, 1887, 1203], 'text': '25°C,', 'line_num': 1, 'block_num': 34, 'span_num': 151}, {'bbox': [1903, 1183, 1977, 1201], 'text': 'AM=1.5', 'line_num': 1, 'block_num': 34, 'span_num': 152}, {'bbox': [1264, 1248, 1349, 1268], 'text': 'Bifacial', 'line_num': 1, 'block_num': 35, 'span_num': 153}, {'bbox': [1357, 1248, 1555, 1274], 'text': 'Output-Backside', 'line_num': 1, 'block_num': 35, 'span_num': 154}, {'bbox': [1564, 1249, 1637, 1268], 'text': 'Power', 'line_num': 1, 'block_num': 35, 'span_num': 155}, {'bbox': [1644, 1248, 1696, 1268], 'text': 'Gain', 'line_num': 1, 'block_num': 35, 'span_num': 156}, {'bbox': [1334, 1296, 1438, 1321], 'text': 'Pmax(W)', 'line_num': 1, 'block_num': 36, 'span_num': 157}, {'bbox': [1733, 1298, 1775, 1317], 'text': '478', 'line_num': 1, 'block_num': 37, 'span_num': 158}, {'bbox': [1854, 1298, 1896, 1317], 'text': '484', 'line_num': 1, 'block_num': 37, 'span_num': 159}, {'bbox': [1975, 1298, 2017, 1317], 'text': '489', 'line_num': 1, 'block_num': 37, 'span_num': 160}, {'bbox': [2100, 1298, 2141, 1317], 'text': '495', 'line_num': 1, 'block_num': 37, 'span_num': 161}, {'bbox': [2222, 1298, 2262, 1317], 'text': 'soo', 'line_num': 1, 'block_num': 37, 'span_num': 162}, {'bbox': [1264, 1324, 1313, 1343], 'text': '10%', 'line_num': 1, 'block_num': 39, 'span_num': 163}, {'bbox': [1334, 1349, 1420, 1369], 'text': 'Module', 'line_num': 2, 'block_num': 39, 'span_num': 164}, {'bbox': [1428, 1349, 1540, 1375], 'text': 'efficiency', 'line_num': 2, 'block_num': 39, 'span_num': 165}, {'bbox': [1547, 1348, 1584, 1371], 'text': '(%)', 'line_num': 2, 'block_num': 39, 'span_num': 166}, {'bbox': [1724, 1350, 1786, 1369], 'text': '21.99', 'line_num': 2, 'block_num': 39, 'span_num': 167}, {'bbox': [1844, 1350, 1906, 1369], 'text': '22.27', 'line_num': 2, 'block_num': 39, 'span_num': 168}, {'bbox': [1966, 1350, 2028, 1369], 'text': '22.50', 'line_num': 2, 'block_num': 39, 'span_num': 169}, {'bbox': [2089, 1350, 2152, 1369], 'text': '22.77', 'line_num': 2, 'block_num': 39, 'span_num': 170}, {'bbox': [2211, 1350, 2273, 1369], 'text': '23.00', 'line_num': 2, 'block_num': 39, 'span_num': 171}, {'bbox': [1334, 1394, 1438, 1429], 'text': 'Pmax(W)', 'line_num': 3, 'block_num': 39, 'span_num': 172}, {'bbox': [1735, 1399, 1775, 1418], 'text': '$22', 'line_num': 3, 'block_num': 39, 'span_num': 173}, {'bbox': [1855, 1399, 1895, 1418], 'text': '528', 'line_num': 3, 'block_num': 39, 'span_num': 174}, {'bbox': [1976, 1399, 2017, 1418], 'text': '534', 'line_num': 3, 'block_num': 39, 'span_num': 175}, {'bbox': [2100, 1399, 2140, 1418], 'text': '540', 'line_num': 3, 'block_num': 39, 'span_num': 176}, {'bbox': [2222, 1399, 2262, 1418], 'text': 'S46', 'line_num': 3, 'block_num': 39, 'span_num': 177}, {'bbox': [1263, 1424, 1313, 1443], 'text': '20%', 'line_num': 4, 'block_num': 39, 'span_num': 178}, {'bbox': [1334, 1450, 1420, 1470], 'text': 'Module', 'line_num': 1, 'block_num': 41, 'span_num': 179}, {'bbox': [1428, 1450, 1540, 1476], 'text': 'efficiency', 'line_num': 1, 'block_num': 41, 'span_num': 180}, {'bbox': [1547, 1449, 1584, 1472], 'text': '(%)', 'line_num': 1, 'block_num': 41, 'span_num': 181}, {'bbox': [1724, 1450, 1786, 1469], 'text': '24.02', 'line_num': 1, 'block_num': 42, 'span_num': 182}, {'bbox': [1844, 1450, 1907, 1469], 'text': '24.29', 'line_num': 1, 'block_num': 42, 'span_num': 183}, {'bbox': [1965, 1450, 2028, 1469], 'text': '24.57', 'line_num': 1, 'block_num': 42, 'span_num': 184}, {'bbox': [2090, 1450, 2152, 1469], 'text': '24.84', 'line_num': 1, 'block_num': 42, 'span_num': 185}, {'bbox': [2211, 1450, 2273, 1469], 'text': '25.12', 'line_num': 1, 'block_num': 42, 'span_num': 186}, {'bbox': [1262, 1550, 1431, 1595], 'text': 'Working', 'line_num': 1, 'block_num': 43, 'span_num': 187}, {'bbox': [1443, 1550, 1743, 1585], 'text': 'Characteristics', 'line_num': 1, 'block_num': 43, 'span_num': 188}, {'bbox': [1267, 1613, 1340, 1632], 'text': 'Power', 'line_num': 1, 'block_num': 50, 'span_num': 189}, {'bbox': [1348, 1612, 1401, 1632], 'text': 'level', 'line_num': 1, 'block_num': 50, 'span_num': 190}, {'bbox': [1750, 1614, 1791, 1633], 'text': '435', 'line_num': 1, 'block_num': 50, 'span_num': 191}, {'bbox': [1867, 1614, 1909, 1633], 'text': '440', 'line_num': 1, 'block_num': 50, 'span_num': 192}, {'bbox': [1982, 1609, 2028, 1644], 'text': '445', 'line_num': 1, 'block_num': 50, 'span_num': 193}, {'bbox': [2105, 1614, 2147, 1633], 'text': '450', 'line_num': 1, 'block_num': 50, 'span_num': 194}, {'bbox': [2223, 1614, 2264, 1633], 'text': '4SS', 'line_num': 1, 'block_num': 50, 'span_num': 195}, {'bbox': [1267, 1666, 1331, 1685], 'text': 'Pmax', 'line_num': 2, 'block_num': 50, 'span_num': 196}, {'bbox': [1337, 1664, 1374, 1687], 'text': '(W)', 'line_num': 2, 'block_num': 50, 'span_num': 197}, {'bbox': [1751, 1666, 1791, 1685], 'text': '323', 'line_num': 2, 'block_num': 50, 'span_num': 198}, {'bbox': [1868, 1666, 1909, 1685], 'text': '327', 'line_num': 2, 'block_num': 50, 'span_num': 199}, {'bbox': [1987, 1666, 2027, 1685], 'text': '330', 'line_num': 2, 'block_num': 50, 'span_num': 200}, {'bbox': [2106, 1666, 2146, 1685], 'text': '334', 'line_num': 2, 'block_num': 50, 'span_num': 201}, {'bbox': [2224, 1666, 2264, 1685], 'text': '337', 'line_num': 2, 'block_num': 50, 'span_num': 202}, {'bbox': [1265, 1718, 1317, 1742], 'text': 'vmp', 'line_num': 3, 'block_num': 50, 'span_num': 203}, {'bbox': [1324, 1715, 1352, 1738], 'text': '(Vv)', 'line_num': 3, 'block_num': 50, 'span_num': 204}, {'bbox': [1740, 1718, 1802, 1737], 'text': '37.83', 'line_num': 3, 'block_num': 50, 'span_num': 205}, {'bbox': [1857, 1718, 1919, 1737], 'text': '38.03', 'line_num': 3, 'block_num': 50, 'span_num': 206}, {'bbox': [1976, 1718, 2037, 1737], 'text': '38.11', 'line_num': 3, 'block_num': 50, 'span_num': 207}, {'bbox': [2095, 1718, 2157, 1737], 'text': '38.35', 'line_num': 3, 'block_num': 50, 'span_num': 208}, {'bbox': [2213, 1718, 2275, 1737], 'text': '38.43', 'line_num': 3, 'block_num': 50, 'span_num': 209}, {'bbox': [1267, 1767, 1305, 1791], 'text': 'Imp', 'line_num': 4, 'block_num': 50, 'span_num': 210}, {'bbox': [1317, 1764, 1347, 1787], 'text': '(A)', 'line_num': 4, 'block_num': 50, 'span_num': 211}, {'bbox': [1747, 1767, 1795, 1786], 'text': '8.54', 'line_num': 4, 'block_num': 50, 'span_num': 212}, {'bbox': [1864, 1767, 1912, 1786], 'text': '8.60', 'line_num': 4, 'block_num': 50, 'span_num': 213}, {'bbox': [1983, 1767, 2031, 1786], 'text': '8.66', 'line_num': 4, 'block_num': 50, 'span_num': 214}, {'bbox': [2102, 1767, 2149, 1786], 'text': '8.71', 'line_num': 4, 'block_num': 50, 'span_num': 215}, {'bbox': [2220, 1767, 2267, 1786], 'text': '8.77', 'line_num': 4, 'block_num': 50, 'span_num': 216}, {'bbox': [1265, 1819, 1306, 1837], 'text': 'Voc', 'line_num': 5, 'block_num': 50, 'span_num': 217}, {'bbox': [1313, 1816, 1342, 1839], 'text': '(V)', 'line_num': 5, 'block_num': 50, 'span_num': 218}, {'bbox': [1739, 1818, 1802, 1837], 'text': '45.60', 'line_num': 5, 'block_num': 50, 'span_num': 219}, {'bbox': [1856, 1818, 1918, 1837], 'text': '45.81', 'line_num': 5, 'block_num': 50, 'span_num': 220}, {'bbox': [1975, 1818, 2038, 1837], 'text': '45.99', 'line_num': 5, 'block_num': 50, 'span_num': 221}, {'bbox': [2094, 1818, 2157, 1837], 'text': '46.19', 'line_num': 5, 'block_num': 50, 'span_num': 222}, {'bbox': [2212, 1818, 2274, 1837], 'text': '46.01', 'line_num': 5, 'block_num': 50, 'span_num': 223}, {'bbox': [1267, 1869, 1296, 1887], 'text': 'Isc', 'line_num': 6, 'block_num': 50, 'span_num': 224}, {'bbox': [1303, 1866, 1332, 1889], 'text': '(A)', 'line_num': 6, 'block_num': 50, 'span_num': 225}, {'bbox': [1747, 1868, 1795, 1887], 'text': '8.96', 'line_num': 6, 'block_num': 50, 'span_num': 226}, {'bbox': [1864, 1868, 1912, 1887], 'text': '9.05', 'line_num': 6, 'block_num': 50, 'span_num': 227}, {'bbox': [1983, 1868, 2031, 1887], 'text': '9.10', 'line_num': 6, 'block_num': 50, 'span_num': 228}, {'bbox': [2102, 1868, 2150, 1887], 'text': '9.16', 'line_num': 6, 'block_num': 50, 'span_num': 229}, {'bbox': [2220, 1868, 2268, 1887], 'text': '9.20', 'line_num': 6, 'block_num': 50, 'span_num': 230}, {'bbox': [1267, 1919, 1340, 1938], 'text': 'Power', 'line_num': 7, 'block_num': 50, 'span_num': 231}, {'bbox': [1347, 1918, 1457, 1938], 'text': 'tolerance', 'line_num': 7, 'block_num': 50, 'span_num': 232}, {'bbox': [1463, 1917, 1500, 1940], 'text': '(%)', 'line_num': 7, 'block_num': 50, 'span_num': 233}, {'bbox': [1980, 1919, 2035, 1938], 'text': '0~+3', 'line_num': 7, 'block_num': 50, 'span_num': 234}, {'bbox': [1267, 1969, 1331, 1988], 'text': 'NOCT', 'line_num': 1, 'block_num': 50, 'span_num': 235}, {'bbox': [1338, 1967, 1377, 1990], 'text': '(°C)', 'line_num': 1, 'block_num': 50, 'span_num': 236}, {'bbox': [1974, 1971, 2040, 1991], 'text': '4442', 'line_num': 1, 'block_num': 50, 'span_num': 237}, {'bbox': [1265, 2014, 1595, 2034], 'text': 'NOCT:Conditions:Irradiance', 'line_num': 1, 'block_num': 51, 'span_num': 238}, {'bbox': [1610, 2012, 1707, 2037], 'text': '800W/m?', 'line_num': 1, 'block_num': 51, 'span_num': 239}, {'bbox': [1718, 2008, 1816, 2043], 'text': 'ambient', 'line_num': 1, 'block_num': 51, 'span_num': 240}, {'bbox': [1824, 2017, 1972, 2039], 'text': 'temperature', 'line_num': 1, 'block_num': 51, 'span_num': 241}, {'bbox': [1981, 2015, 2040, 2036], 'text': '20°C,', 'line_num': 1, 'block_num': 51, 'span_num': 242}, {'bbox': [2054, 2014, 2101, 2034], 'text': 'wind', 'line_num': 1, 'block_num': 51, 'span_num': 243}, {'bbox': [2112, 2014, 2180, 2039], 'text': 'speed', 'line_num': 1, 'block_num': 51, 'span_num': 244}, {'bbox': [2190, 2015, 2246, 2037], 'text': '1m/s', 'line_num': 1, 'block_num': 51, 'span_num': 245}, {'bbox': [1260, 2111, 1522, 2146], 'text': 'Mechanical', 'line_num': 1, 'block_num': 52, 'span_num': 246}, {'bbox': [1580, 2111, 1799, 2146], 'text': 'Characteristics', 'line_num': 1, 'block_num': 52, 'span_num': 247}, {'bbox': [1260, 2174, 1356, 2194], 'text': 'Number', 'line_num': 1, 'block_num': 55, 'span_num': 248}, {'bbox': [1362, 2174, 1386, 2194], 'text': 'of', 'line_num': 1, 'block_num': 55, 'span_num': 249}, {'bbox': [1392, 2174, 1443, 2194], 'text': 'cells', 'line_num': 1, 'block_num': 55, 'span_num': 250}, {'bbox': [1749, 2175, 1829, 2200], 'text': '144pcs', 'line_num': 1, 'block_num': 55, 'span_num': 251}, {'bbox': [1260, 2224, 1306, 2244], 'text': 'Size', 'line_num': 2, 'block_num': 55, 'span_num': 252}, {'bbox': [1313, 2224, 1337, 2244], 'text': 'of', 'line_num': 2, 'block_num': 55, 'span_num': 253}, {'bbox': [1343, 2224, 1382, 2244], 'text': 'cell', 'line_num': 2, 'block_num': 55, 'span_num': 254}, {'bbox': [1389, 2223, 1450, 2246], 'text': '(mm)', 'line_num': 2, 'block_num': 55, 'span_num': 255}, {'bbox': [1749, 2225, 1829, 2244], 'text': '166°83', 'line_num': 2, 'block_num': 55, 'span_num': 256}, {'bbox': [1259, 2277, 1315, 2302], 'text': 'Type', 'line_num': 3, 'block_num': 55, 'span_num': 257}, {'bbox': [1323, 2276, 1346, 2296], 'text': 'of', 'line_num': 3, 'block_num': 55, 'span_num': 258}, {'bbox': [1352, 2276, 1391, 2296], 'text': 'cell', 'line_num': 3, 'block_num': 55, 'span_num': 259}, {'bbox': [1749, 2276, 1812, 2295], 'text': 'Mono', 'line_num': 3, 'block_num': 55, 'span_num': 260}, {'bbox': [1255, 2324, 1372, 2344], 'text': 'Thickness', 'line_num': 4, 'block_num': 55, 'span_num': 261}, {'bbox': [1380, 2324, 1403, 2344], 'text': 'of', 'line_num': 4, 'block_num': 55, 'span_num': 262}, {'bbox': [1409, 2324, 1467, 2350], 'text': 'glass', 'line_num': 4, 'block_num': 55, 'span_num': 263}, {'bbox': [1474, 2323, 1535, 2346], 'text': '(mm)', 'line_num': 4, 'block_num': 55, 'span_num': 264}, {'bbox': [1749, 2324, 1783, 2343], 'text': '2.0', 'line_num': 4, 'block_num': 55, 'span_num': 265}, {'bbox': [1262, 2378, 1317, 2403], 'text': 'Type', 'line_num': 1, 'block_num': 56, 'span_num': 266}, {'bbox': [1325, 2377, 1349, 2397], 'text': 'of', 'line_num': 1, 'block_num': 56, 'span_num': 267}, {'bbox': [1355, 2377, 1423, 2397], 'text': 'frame', 'line_num': 1, 'block_num': 56, 'span_num': 268}, {'bbox': [1746, 2376, 1856, 2396], 'text': 'Anodized', 'line_num': 1, 'block_num': 57, 'span_num': 269}, {'bbox': [1864, 2376, 1984, 2396], 'text': 'aluminum', 'line_num': 1, 'block_num': 57, 'span_num': 270}, {'bbox': [1992, 2376, 2049, 2402], 'text': 'alloy', 'line_num': 1, 'block_num': 57, 'span_num': 271}, {'bbox': [1262, 2426, 1365, 2446], 'text': 'Junction', 'line_num': 1, 'block_num': 60, 'span_num': 272}, {'bbox': [1374, 2426, 1414, 2446], 'text': 'box', 'line_num': 1, 'block_num': 60, 'span_num': 273}, {'bbox': [1749, 2426, 1798, 2445], 'text': 'IP68', 'line_num': 1, 'block_num': 60, 'span_num': 274}, {'bbox': [1262, 2478, 1308, 2498], 'text': 'Size', 'line_num': 2, 'block_num': 60, 'span_num': 275}, {'bbox': [1315, 2478, 1339, 2498], 'text': 'of', 'line_num': 2, 'block_num': 60, 'span_num': 276}, {'bbox': [1357, 2478, 1435, 2498], 'text': 'module', 'line_num': 2, 'block_num': 60, 'span_num': 277}, {'bbox': [1444, 2477, 1502, 2500], 'text': '(mm)', 'line_num': 2, 'block_num': 60, 'span_num': 278}, {'bbox': [1748, 2479, 1912, 2498], 'text': '2094°1038*°30', 'line_num': 2, 'block_num': 60, 'span_num': 279}, {'bbox': [1261, 2524, 1332, 2560], 'text': 'Weight', 'line_num': 3, 'block_num': 60, 'span_num': 280}, {'bbox': [1347, 2528, 1392, 2555], 'text': '(kg)', 'line_num': 3, 'block_num': 60, 'span_num': 281}, {'bbox': [1749, 2531, 1797, 2550], 'text': '27.5', 'line_num': 3, 'block_num': 60, 'span_num': 282}, {'bbox': [1259, 2577, 1480, 2601], 'text': 'Cables/connectors', 'line_num': 1, 'block_num': 61, 'span_num': 283}, {'bbox': [1748, 2577, 1882, 2601], 'text': '4mm?,MC4', 'line_num': 1, 'block_num': 62, 'span_num': 284}, {'bbox': [1890, 2579, 2022, 2605], 'text': 'compatible', 'line_num': 1, 'block_num': 62, 'span_num': 285}, {'bbox': [1260, 2629, 1340, 2655], 'text': 'Length', 'line_num': 1, 'block_num': 64, 'span_num': 286}, {'bbox': [1348, 2629, 1371, 2649], 'text': 'of', 'line_num': 1, 'block_num': 64, 'span_num': 287}, {'bbox': [1378, 2629, 1443, 2649], 'text': 'Cabel', 'line_num': 1, 'block_num': 64, 'span_num': 288}, {'bbox': [1749, 2624, 1847, 2656], 'text': 'Portrait:', 'line_num': 1, 'block_num': 65, 'span_num': 289}, {'bbox': [1857, 2629, 2062, 2652], 'text': '+300mm/-300mm', 'line_num': 1, 'block_num': 65, 'span_num': 290}, {'bbox': [1261, 2730, 1456, 2765], 'text': 'Maximum', 'line_num': 1, 'block_num': 66, 'span_num': 291}, {'bbox': [1473, 2730, 1620, 2775], 'text': 'Ratings', 'line_num': 1, 'block_num': 66, 'span_num': 292}, {'bbox': [1258, 2794, 1376, 2820], 'text': 'Operating', 'line_num': 1, 'block_num': 67, 'span_num': 293}, {'bbox': [1383, 2794, 1577, 2820], 'text': 'Temperature(°C)', 'line_num': 1, 'block_num': 67, 'span_num': 294}, {'bbox': [1258, 2844, 1376, 2870], 'text': 'Operating', 'line_num': 2, 'block_num': 67, 'span_num': 295}, {'bbox': [1384, 2844, 1531, 2870], 'text': 'humidity(°C)', 'line_num': 2, 'block_num': 67, 'span_num': 296}, {'bbox': [1749, 2790, 1827, 2809], 'text': '-40~85', 'line_num': 1, 'block_num': 68, 'span_num': 297}, {'bbox': [1749, 2846, 1804, 2865], 'text': 'S~8S', 'line_num': 2, 'block_num': 68, 'span_num': 298}, {'bbox': [1257, 2894, 1374, 2914], 'text': 'Allowable', 'line_num': 1, 'block_num': 70, 'span_num': 299}, {'bbox': [1383, 2894, 1427, 2914], 'text': 'Hail', 'line_num': 1, 'block_num': 70, 'span_num': 300}, {'bbox': [1436, 2894, 1491, 2914], 'text': 'Load', 'line_num': 1, 'block_num': 70, 'span_num': 301}, {'bbox': [1749, 2897, 1822, 2915], 'text': '25mm', 'line_num': 1, 'block_num': 72, 'span_num': 302}, {'bbox': [1831, 2895, 1914, 2915], 'text': 'ice-ball', 'line_num': 1, 'block_num': 72, 'span_num': 303}, {'bbox': [1922, 2895, 1973, 2915], 'text': 'with', 'line_num': 1, 'block_num': 72, 'span_num': 304}, {'bbox': [1981, 2895, 2074, 2921], 'text': 'velocity', 'line_num': 1, 'block_num': 72, 'span_num': 305}, {'bbox': [2081, 2895, 2104, 2915], 'text': 'of', 'line_num': 1, 'block_num': 72, 'span_num': 306}, {'bbox': [2111, 2896, 2182, 2919], 'text': '23m/s', 'line_num': 1, 'block_num': 72, 'span_num': 307}, {'bbox': [1809, 3066, 1851, 3086], 'text': 'Tel:', 'line_num': 1, 'block_num': 73, 'span_num': 308}, {'bbox': [1857, 3067, 2057, 3086], 'text': '86-519-82585880', 'line_num': 1, 'block_num': 73, 'span_num': 309}, {'bbox': [1809, 3103, 1854, 3129], 'text': 'Zip:', 'line_num': 1, 'block_num': 73, 'span_num': 310}, {'bbox': [1862, 3104, 1943, 3123], 'text': '213213', 'line_num': 1, 'block_num': 73, 'span_num': 311}, {'bbox': [1808, 3140, 2016, 3164], 'text': 'Add:No.18,Jinwu', 'line_num': 1, 'block_num': 73, 'span_num': 312}, {'bbox': [2025, 3136, 2083, 3168], 'text': 'Road,', 'line_num': 1, 'block_num': 73, 'span_num': 313}, {'bbox': [2097, 3136, 2159, 3168], 'text': 'Jintan', 'line_num': 1, 'block_num': 73, 'span_num': 314}, {'bbox': [2175, 3140, 2225, 3164], 'text': 'Dist,', 'line_num': 1, 'block_num': 73, 'span_num': 315}, {'bbox': [1809, 3177, 1941, 3203], 'text': 'Changzhou', 'line_num': 2, 'block_num': 73, 'span_num': 316}, {'bbox': [1941, 3173, 2001, 3207], 'text': 'City,', 'line_num': 2, 'block_num': 73, 'span_num': 317}, {'bbox': [2006, 3177, 2092, 3203], 'text': 'Jiangsu', 'line_num': 2, 'block_num': 73, 'span_num': 318}, {'bbox': [2103, 3177, 2211, 3197], 'text': 'Province.', 'line_num': 2, 'block_num': 73, 'span_num': 319}, {'bbox': [1810, 3210, 1884, 3244], 'text': 'Email:', 'line_num': 3, 'block_num': 73, 'span_num': 320}, {'bbox': [1890, 3214, 2187, 3240], 'text': 'marketing@egingpv.com', 'line_num': 3, 'block_num': 73, 'span_num': 321}, {'bbox': [1809, 3251, 1868, 3271], 'text': 'Web:', 'line_num': 4, 'block_num': 73, 'span_num': 322}, {'bbox': [1874, 3251, 2091, 3277], 'text': 'www.egingpv.com', 'line_num': 4, 'block_num': 73, 'span_num': 323}]
        >>> objects = [{'label': 'table', 'score': 0.9971179962158203, 'bbox': [1202.9041748046875, 2063.8251953125, 2317.14306640625, 2670.127685546875]}, {'label': 'table', 'score': 0.9645423293113708, 'bbox': [1272.7747802734375, 2700.2958984375, 2321.947265625, 2941.731689453125]}, {'label': 'table', 'score': 0.9867339134216309, 'bbox': [126.97383117675781, 2710.196533203125, 1138.2745361328125, 3049.474365234375]}, {'label': 'table', 'score': 0.9809158444404602, 'bbox': [1209.289794921875, 1211.23828125, 2373.331787109375, 1507.68310546875]}, {'label': 'table', 'score': 0.5817432403564453, 'bbox': [1188.874755859375, 1328.92822265625, 2364.0029296875, 1663.665283203125]}, {'label': 'table', 'score': 0.9760532975196838, 'bbox': [1192.669677734375, 1524.52001953125, 2356.83251953125, 2041.1297607421875]}, {'label': 'table', 'score': 0.9982981085777283, 'bbox': [1227.2569580078125, 491.8607177734375, 2349.684326171875, 1187.0966796875]}]
        >>> class_thresholds = {'no object': 10, 'table': 0.8, 'table rotated': 0.7}
        >>> crops = objects_to_crops(img, tokens, objects, class_thresholds)
        >>> crops[0]['tokens']
        [{'bbox': [67.0958251953125, 57.1748046875, 329.0958251953125, 92.1748046875], 'text': 'Mechanical', 'line_num': 1, 'block_num': 52, 'span_num': 246}, {'bbox': [387.0958251953125, 57.1748046875, 606.0958251953125, 92.1748046875], 'text': 'Characteristics', 'line_num': 1, 'block_num': 52, 'span_num': 247}, {'bbox': [67.0958251953125, 120.1748046875, 163.0958251953125, 140.1748046875], 'text': 'Number', 'line_num': 1, 'block_num': 55, 'span_num': 248}, {'bbox': [169.0958251953125, 120.1748046875, 193.0958251953125, 140.1748046875], 'text': 'of', 'line_num': 1, 'block_num': 55, 'span_num': 249}, {'bbox': [199.0958251953125, 120.1748046875, 250.0958251953125, 140.1748046875], 'text': 'cells', 'line_num': 1, 'block_num': 55, 'span_num': 250}, {'bbox': [556.0958251953125, 121.1748046875, 636.0958251953125, 146.1748046875], 'text': '144pcs', 'line_num': 1, 'block_num': 55, 'span_num': 251}, {'bbox': [67.0958251953125, 170.1748046875, 113.0958251953125, 190.1748046875], 'text': 'Size', 'line_num': 2, 'block_num': 55, 'span_num': 252}, {'bbox': [120.0958251953125, 170.1748046875, 144.0958251953125, 190.1748046875], 'text': 'of', 'line_num': 2, 'block_num': 55, 'span_num': 253}, {'bbox': [150.0958251953125, 170.1748046875, 189.0958251953125, 190.1748046875], 'text': 'cell', 'line_num': 2, 'block_num': 55, 'span_num': 254}, {'bbox': [196.0958251953125, 169.1748046875, 257.0958251953125, 192.1748046875], 'text': '(mm)', 'line_num': 2, 'block_num': 55, 'span_num': 255}, {'bbox': [556.0958251953125, 171.1748046875, 636.0958251953125, 190.1748046875], 'text': '166°83', 'line_num': 2, 'block_num': 55, 'span_num': 256}, {'bbox': [66.0958251953125, 223.1748046875, 122.0958251953125, 248.1748046875], 'text': 'Type', 'line_num': 3, 'block_num': 55, 'span_num': 257}, {'bbox': [130.0958251953125, 222.1748046875, 153.0958251953125, 242.1748046875], 'text': 'of', 'line_num': 3, 'block_num': 55, 'span_num': 258}, {'bbox': [159.0958251953125, 222.1748046875, 198.0958251953125, 242.1748046875], 'text': 'cell', 'line_num': 3, 'block_num': 55, 'span_num': 259}, {'bbox': [556.0958251953125, 222.1748046875, 619.0958251953125, 241.1748046875], 'text': 'Mono', 'line_num': 3, 'block_num': 55, 'span_num': 260}, {'bbox': [62.0958251953125, 270.1748046875, 179.0958251953125, 290.1748046875], 'text': 'Thickness', 'line_num': 4, 'block_num': 55, 'span_num': 261}, {'bbox': [187.0958251953125, 270.1748046875, 210.0958251953125, 290.1748046875], 'text': 'of', 'line_num': 4, 'block_num': 55, 'span_num': 262}, {'bbox': [216.0958251953125, 270.1748046875, 274.0958251953125, 296.1748046875], 'text': 'glass', 'line_num': 4, 'block_num': 55, 'span_num': 263}, {'bbox': [281.0958251953125, 269.1748046875, 342.0958251953125, 292.1748046875], 'text': '(mm)', 'line_num': 4, 'block_num': 55, 'span_num': 264}, {'bbox': [556.0958251953125, 270.1748046875, 590.0958251953125, 289.1748046875], 'text': '2.0', 'line_num': 4, 'block_num': 55, 'span_num': 265}, {'bbox': [69.0958251953125, 324.1748046875, 124.0958251953125, 349.1748046875], 'text': 'Type', 'line_num': 1, 'block_num': 56, 'span_num': 266}, {'bbox': [132.0958251953125, 323.1748046875, 156.0958251953125, 343.1748046875], 'text': 'of', 'line_num': 1, 'block_num': 56, 'span_num': 267}, {'bbox': [162.0958251953125, 323.1748046875, 230.0958251953125, 343.1748046875], 'text': 'frame', 'line_num': 1, 'block_num': 56, 'span_num': 268}, {'bbox': [553.0958251953125, 322.1748046875, 663.0958251953125, 342.1748046875], 'text': 'Anodized', 'line_num': 1, 'block_num': 57, 'span_num': 269}, {'bbox': [671.0958251953125, 322.1748046875, 791.0958251953125, 342.1748046875], 'text': 'aluminum', 'line_num': 1, 'block_num': 57, 'span_num': 270}, {'bbox': [799.0958251953125, 322.1748046875, 856.0958251953125, 348.1748046875], 'text': 'alloy', 'line_num': 1, 'block_num': 57, 'span_num': 271}, {'bbox': [69.0958251953125, 372.1748046875, 172.0958251953125, 392.1748046875], 'text': 'Junction', 'line_num': 1, 'block_num': 60, 'span_num': 272}, {'bbox': [181.0958251953125, 372.1748046875, 221.0958251953125, 392.1748046875], 'text': 'box', 'line_num': 1, 'block_num': 60, 'span_num': 273}, {'bbox': [556.0958251953125, 372.1748046875, 605.0958251953125, 391.1748046875], 'text': 'IP68', 'line_num': 1, 'block_num': 60, 'span_num': 274}, {'bbox': [69.0958251953125, 424.1748046875, 115.0958251953125, 444.1748046875], 'text': 'Size', 'line_num': 2, 'block_num': 60, 'span_num': 275}, {'bbox': [122.0958251953125, 424.1748046875, 146.0958251953125, 444.1748046875], 'text': 'of', 'line_num': 2, 'block_num': 60, 'span_num': 276}, {'bbox': [164.0958251953125, 424.1748046875, 242.0958251953125, 444.1748046875], 'text': 'module', 'line_num': 2, 'block_num': 60, 'span_num': 277}, {'bbox': [251.0958251953125, 423.1748046875, 309.0958251953125, 446.1748046875], 'text': '(mm)', 'line_num': 2, 'block_num': 60, 'span_num': 278}, {'bbox': [555.0958251953125, 425.1748046875, 719.0958251953125, 444.1748046875], 'text': '2094°1038*°30', 'line_num': 2, 'block_num': 60, 'span_num': 279}, {'bbox': [68.0958251953125, 470.1748046875, 139.0958251953125, 506.1748046875], 'text': 'Weight', 'line_num': 3, 'block_num': 60, 'span_num': 280}, {'bbox': [154.0958251953125, 474.1748046875, 199.0958251953125, 501.1748046875], 'text': '(kg)', 'line_num': 3, 'block_num': 60, 'span_num': 281}, {'bbox': [556.0958251953125, 477.1748046875, 604.0958251953125, 496.1748046875], 'text': '27.5', 'line_num': 3, 'block_num': 60, 'span_num': 282}, {'bbox': [66.0958251953125, 523.1748046875, 287.0958251953125, 547.1748046875], 'text': 'Cables/connectors', 'line_num': 1, 'block_num': 61, 'span_num': 283}, {'bbox': [555.0958251953125, 523.1748046875, 689.0958251953125, 547.1748046875], 'text': '4mm?,MC4', 'line_num': 1, 'block_num': 62, 'span_num': 284}, {'bbox': [697.0958251953125, 525.1748046875, 829.0958251953125, 551.1748046875], 'text': 'compatible', 'line_num': 1, 'block_num': 62, 'span_num': 285}, {'bbox': [67.0958251953125, 575.1748046875, 147.0958251953125, 601.1748046875], 'text': 'Length', 'line_num': 1, 'block_num': 64, 'span_num': 286}, {'bbox': [155.0958251953125, 575.1748046875, 178.0958251953125, 595.1748046875], 'text': 'of', 'line_num': 1, 'block_num': 64, 'span_num': 287}, {'bbox': [185.0958251953125, 575.1748046875, 250.0958251953125, 595.1748046875], 'text': 'Cabel', 'line_num': 1, 'block_num': 64, 'span_num': 288}, {'bbox': [556.0958251953125, 570.1748046875, 654.0958251953125, 602.1748046875], 'text': 'Portrait:', 'line_num': 1, 'block_num': 65, 'span_num': 289}, {'bbox': [664.0958251953125, 575.1748046875, 869.0958251953125, 598.1748046875], 'text': '+300mm/-300mm', 'line_num': 1, 'block_num': 65, 'span_num': 290}]
    """

    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        cropped_table = {}

        bbox = obj['bbox']
        bbox = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding]

        cropped_img = img.crop(bbox)

        # Applying image filters to enhance quality
        cropped_img = cropped_img.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Contrast(cropped_img)
        cropped_img = enhancer.enhance(2)

        # Scaling the cropped image up for better resolution
        #scale_factor = 2
        #width, height = cropped_img.size
        #cropped_img = cropped_img.resize((width * scale_factor, height * scale_factor), Image.LANCZOS)

        table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5]
        for token in table_tokens:
            token['bbox'] = [token['bbox'][0] - bbox[0],
                             token['bbox'][1] - bbox[1],
                             token['bbox'][2] - bbox[0],
                             token['bbox'][3] - bbox[1]]

        # If table is predicted to be rotated, rotate cropped image and tokens/words:
        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
            for token in table_tokens:
                bbox = token['bbox']
                bbox = [cropped_img.size[0] - bbox[3] - 1,
                        bbox[0],
                        cropped_img.size[0] - bbox[1] - 1,
                        bbox[2]]
                token['bbox'] = bbox

        cropped_table['image'] = cropped_img
        cropped_table['tokens'] = table_tokens

        table_crops.append(cropped_table)

    return table_crops


def objects_to_structures(objects, tokens, class_thresholds):
    """
    Process the bounding boxes produced by the table structure recognition model into
    a *consistent* set of table structures (rows, columns, spanning cells, headers).
    This entails resolving conflicts/overlaps, and ensuring the boxes meet certain alignment
    conditions (for example: rows should all have the same width, etc.).
    """

    tables = [obj for obj in objects if obj['label'] == 'table']
    table_structures = []

    for table in tables:
        table_objects = [obj for obj in objects if iob(obj['bbox'], table['bbox']) >= 0.5]
        table_tokens = [token for token in tokens if iob(token['bbox'], table['bbox']) >= 0.5]

        structure = {}

        # Detect table name and position it appropriately
        table_name = None
        for obj in table_objects:
            if obj['label'] == 'table name':
                table_name = obj
                break

        if table_name:
            table_name_rect = Rect(table_name['bbox'])

            # Check if the table name is part of the first row
            first_row = None
            for row in table_objects:
                if row['label'] == 'table row':
                    first_row_rect = Rect(row['bbox'])
                    if table_name_rect.intersect(first_row_rect).get_area() > 0:
                        first_row = row
                        break

            if first_row:
                # Check if the first row contains only one cell, and the table name spans across it
                if first_row_rect == table_name_rect:
                    structure['table name'] = {
                        'bbox': table_name['bbox'],
                        'position': 'within_row'
                    }
                    table_objects.remove(first_row)  # Remove the first row if it's the table name
                else:
                    structure['table name'] = {
                        'bbox': table_name['bbox'],
                        'position': 'above_row'
                    }
            else:
                structure['table name'] = {
                    'bbox': table_name['bbox'],
                    'position': 'above_row'
                }

        #table_name = [obj for obj in table_objects if obj['label'] == 'table name']
        columns = [obj for obj in table_objects if obj['label'] == 'table column']
        rows = [obj for obj in table_objects if obj['label'] == 'table row']
        column_headers = [obj for obj in table_objects if obj['label'] == 'table column header']
        row_headers = [obj for obj in table_objects if obj['label'] == 'table row header']

        spanning_cells = [obj for obj in table_objects if obj['label'] == 'table spanning cell']
        for obj in spanning_cells:
            obj['projected row header'] = False
            obj['projected column header'] = False

        projected_row_headers = [obj for obj in table_objects if obj['label'] == 'table projected row header']
        for obj in projected_row_headers:
            obj['projected row header'] = True
            obj['projected column header'] = False
        spanning_cells += projected_row_headers

        projected_column_headers = [obj for obj in table_objects if obj['label'] == 'table projected column header']
        for obj in projected_column_headers:
            obj['projected column header'] = True
            obj['projected row header'] = False
        spanning_cells += projected_column_headers

        for obj in rows:
            obj['column header'] = False
            for header_obj in column_headers:
                if iob(obj['bbox'], header_obj['bbox']) >= 0.5:
                    obj['column header'] = True

        for obj in columns:
            obj['row header'] = False
            for header_obj in row_headers:
                if iob(obj['bbox'], header_obj['bbox']) >= 0.5:
                    obj['row header'] = True

        # Refine table structures
        rows = postprocess.refine_rows(rows, table_tokens, class_thresholds['table row'])
        columns = postprocess.refine_columns(columns, table_tokens, class_thresholds['table column'])

        # Shrink table bbox to just the total height of the rows
        # and the total width of the columns
        row_rect = Rect()
        for obj in rows:
            row_rect.include_rect(obj['bbox'])
        column_rect = Rect()
        for obj in columns:
            column_rect.include_rect(obj['bbox'])
        table['row_column_bbox'] = [column_rect[0], row_rect[1], column_rect[2], row_rect[3]]
        table['bbox'] = table['row_column_bbox']

        # Process the rows and columns into a complete segmented table
        columns = postprocess.align_columns(columns, table['row_column_bbox'])
        rows = postprocess.align_rows(rows, table['row_column_bbox'])

        #structure['table name'] = table_name
        structure['rows'] = rows
        structure['columns'] = columns
        structure['column headers'] = column_headers
        structure['row headers'] = row_headers
        structure['spanning cells'] = spanning_cells

        # Add check to handle tables with both row and column headers
        if len(row_headers) > 0 and len(column_headers) > 0:
            structure['both headers'] = True
        else:
            structure['both headers'] = False

        if len(rows) > 0 and len(columns) > 1:
            structure = refine_table_structure(structure, class_thresholds)

        table_structures.append(structure)

    return table_structures

def structure_to_cells(table_structure, tokens):
    """
    Assuming the row, column, spanning cell, and header bounding boxes have
    been refined into a set of consistent table structures, process these
    table structures into table cells. This is a universal representation
    format for the table, which can later be exported to Pandas or CSV formats.
    Classify the cells as header/access cells or data cells
    based on if they intersect with the header bounding box.
    """
    #table_name_header = table_structure['table name']
    table_name = table_structure.get('table name', None)
    #table_name = table_structure['table name']
    columns = table_structure['columns']
    rows = table_structure['rows']
    spanning_cells = table_structure['spanning cells']
    cells = []
    subcells = []

    # Identify complete cells and subcells
    for column_num, column in enumerate(columns):
        for row_num, row in enumerate(rows):
            column_rect = Rect(list(column['bbox']))
            row_rect = Rect(list(row['bbox']))
            cell_rect = row_rect.intersect(column_rect)
            #table_name_header = 'table name' in row and row['table name']
            row_header = 'row header' in column and column['row header']
            column_header = 'column header' in row and row['column header']

            # Handle the table name when it's part of the first row
            table_name_flag = False
            if table_name and table_name['position'] == 'within_row' and row_num == 0:
                table_name_flag = True
                if Rect(table_name['bbox']) == cell_rect:
                    # Skip adding this as a regular cell
                    continue

            cell = {'bbox': list(cell_rect), 'column_nums': [column_num], 'row_nums': [row_num],
                    'column header': column_header, 'row header': row_header, 'table name':table_name_flag}


            cell['subcell'] = False
            for spanning_cell in spanning_cells:
                spanning_cell_rect = Rect(list(spanning_cell['bbox']))
                if (spanning_cell_rect.intersect(cell_rect).get_area()
                    / cell_rect.get_area()) > 0.5:
                    cell['subcell'] = True
                    break

            if cell['subcell']:
                subcells.append(cell)
            else:
                # cell text = extract_text_inside_bbox(table_spans, cell['bbox'])
                # cell['cell text'] = cell text
                cell['projected row header'] = False
                cell['projected column header'] = False
                cells.append(cell)

    for spanning_cell in spanning_cells:
        spanning_cell_rect = Rect(list(spanning_cell['bbox']))
        cell_columns = set()
        cell_rows = set()
        cell_rect = None
        column_header = True
        row_header = True
        for subcell in subcells:
            subcell_rect = Rect(list(subcell['bbox']))
            subcell_area = subcell_rect.get_area()
            if subcell_area > 0:
                if (subcell_rect.intersect(spanning_cell_rect).get_area() / subcell_area) > 0.5:
                    if cell_rect is None:
                        cell_rect = Rect(list(subcell['bbox']))
                    else:
                        cell_rect.include_rect(Rect(list(subcell['bbox'])))
                    cell_rows = cell_rows.union(set(subcell['row_nums']))
                    cell_columns = cell_columns.union(set(subcell['column_nums']))
                    # By convention here, all subcells must be classified
                    # as header cells for a spanning cell to be classified as a header cell;
                    # otherwise, this could lead to a non-rectangular header region
                    column_header = column_header and subcell.get('column header', False)
                    row_header = row_header and subcell.get('row header', False)
        if len(cell_rows) > 0 and len(cell_columns) > 0:
            cell = {'bbox': list(cell_rect), 'column_nums': list(cell_columns), 'row_nums': list(cell_rows),
                    'column header': column_header, 'row header': row_header,
                    'projected row header': spanning_cell['projected row header'],
                    'projected column header': spanning_cell['projected column header'],
                    'table name':table_name}
            cells.append(cell)

    # Compute a confidence score based on how well the page tokens
    # slot into the cells reported by the model
    _, _, cell_match_scores = postprocess.slot_into_containers(cells, tokens)
    try:
        mean_match_score = sum(cell_match_scores) / len(cell_match_scores)
        min_match_score = min(cell_match_scores)
        confidence_score = (mean_match_score + min_match_score) / 2
    except:
        confidence_score = 0

    # Dilate rows and columns before final extraction
    # dilated_columns = fill_column_gaps(columns, table_bbox)
    dilated_columns = columns
    # dilated_rows = fill_row_gaps(rows, table_bbox)
    dilated_rows = rows
    for cell in cells:
        column_rect = Rect()
        for column_num in cell['column_nums']:
            column_rect.include_rect(list(dilated_columns[column_num]['bbox']))
        row_rect = Rect()
        for row_num in cell['row_nums']:
            row_rect.include_rect(list(dilated_rows[row_num]['bbox']))
        cell_rect = column_rect.intersect(row_rect)
        cell['bbox'] = list(cell_rect)

    span_nums_by_cell, _, _ = postprocess.slot_into_containers(cells, tokens, overlap_threshold=0.001,
                                                               unique_assignment=True, forced_assignment=False)

    for cell, cell_span_nums in zip(cells, span_nums_by_cell):
        cell_spans = [tokens[num] for num in cell_span_nums]
        # TODO: Refine how text is extracted; should be character-based, not span-based;
        # but need to associate
        cell['cell text'] = postprocess.extract_text_from_spans(cell_spans, remove_integer_superscripts=False)
        cell['spans'] = cell_spans

    # Adjust the row, column, and cell bounding boxes to reflect the extracted text
    num_rows = len(rows)
    rows = postprocess.sort_objects_top_to_bottom(rows)
    num_columns = len(columns)
    columns = postprocess.sort_objects_left_to_right(columns)
    min_y_values_by_row = defaultdict(list)
    max_y_values_by_row = defaultdict(list)
    min_x_values_by_column = defaultdict(list)
    max_x_values_by_column = defaultdict(list)
    for cell in cells:
        min_row = min(cell["row_nums"])
        max_row = max(cell["row_nums"])
        min_column = min(cell["column_nums"])
        max_column = max(cell["column_nums"])
        for span in cell['spans']:
            min_x_values_by_column[min_column].append(span['bbox'][0])
            min_y_values_by_row[min_row].append(span['bbox'][1])
            max_x_values_by_column[max_column].append(span['bbox'][2])
            max_y_values_by_row[max_row].append(span['bbox'][3])
    for row_num, row in enumerate(rows):
        if len(min_x_values_by_column[0]) > 0:
            row['bbox'][0] = min(min_x_values_by_column[0])
        if len(min_y_values_by_row[row_num]) > 0:
            row['bbox'][1] = min(min_y_values_by_row[row_num])
        if len(max_x_values_by_column[num_columns - 1]) > 0:
            row['bbox'][2] = max(max_x_values_by_column[num_columns - 1])
        if len(max_y_values_by_row[row_num]) > 0:
            row['bbox'][3] = max(max_y_values_by_row[row_num])
    for column_num, column in enumerate(columns):
        if len(min_x_values_by_column[column_num]) > 0:
            column['bbox'][0] = min(min_x_values_by_column[column_num])
        if len(min_y_values_by_row[0]) > 0:
            column['bbox'][1] = min(min_y_values_by_row[0])
        if len(max_x_values_by_column[column_num]) > 0:
            column['bbox'][2] = max(max_x_values_by_column[column_num])
        if len(max_y_values_by_row[num_rows - 1]) > 0:
            column['bbox'][3] = max(max_y_values_by_row[num_rows - 1])
    for cell in cells:
        row_rect = Rect()
        column_rect = Rect()
        for row_num in cell['row_nums']:
            row_rect.include_rect(list(rows[row_num]['bbox']))
        for column_num in cell['column_nums']:
            column_rect.include_rect(list(columns[column_num]['bbox']))
        cell_rect = row_rect.intersect(column_rect)
        if cell_rect.get_area() > 0:
            cell['bbox'] = list(cell_rect)
            pass

    return cells, confidence_score

def cells_to_csv(cells):
    if len(cells) > 0:
        num_columns = max([max(cell['column_nums']) for cell in cells]) + 1
        num_rows = max([max(cell['row_nums']) for cell in cells]) + 1
    else:
        return

    header_cells = [cell for cell in cells if cell['column header']]
    if len(header_cells) > 0:
        max_header_row = max([max(cell['row_nums']) for cell in header_cells])
    else:
        max_header_row = -1

    table_array = np.empty([num_rows, num_columns], dtype="object")
    if len(cells) > 0:
        for cell in cells:
            for row_num in cell['row_nums']:
                for column_num in cell['column_nums']:
                    table_array[row_num, column_num] = cell["cell text"]

    header = table_array[:max_header_row + 1, :]
    flattened_header = []
    for col in header.transpose():
        flattened_header.append(' | '.join(OrderedDict.fromkeys(col)))
    df = pd.DataFrame(table_array[max_header_row + 1:, :], index=None, columns=flattened_header)
    #df.to_excel('ex.xlsx', index=None)

    return df.to_csv(index=None)

def cells_to_html(cells):
    cells = sorted(cells, key=lambda k: min(k['column_nums']))
    cells = sorted(cells, key=lambda k: min(k['row_nums']))

    table = ET.Element("table")
    current_row = -1

    for cell in cells:
        this_row = min(cell['row_nums'])

        attrib = {}
        colspan = len(cell['column_nums'])
        if colspan > 1:
            attrib['colspan'] = str(colspan)
        rowspan = len(cell['row_nums'])
        if rowspan > 1:
            attrib['rowspan'] = str(rowspan)
        if this_row > current_row:
            current_row = this_row
            if cell['column header']:
                cell_tag = "th"
                row = ET.SubElement(table, "thead")
            else:
                cell_tag = "td"
                row = ET.SubElement(table, "tr")
        tcell = ET.SubElement(row, cell_tag, attrib=attrib)
        tcell.text = cell['cell text']

    return str(ET.tostring(table, encoding="unicode", short_empty_elements=False))

def visualize_detected_tables(img, det_tables, out_path):
    plt.imshow(img, interpolation="lanczos")
    plt.gcf().set_size_inches(20, 20)
    ax = plt.gca()

    for det_table in det_tables:
        bbox = det_table['bbox']

        if det_table['label'] == 'table':
            facecolor = (1, 0, 0.45)
            edgecolor = (1, 0, 0.45)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        elif det_table['label'] == 'table rotated':
            facecolor = (0.95, 0.6, 0.1)
            edgecolor = (0.95, 0.6, 0.1)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        else:
            continue

        rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=linewidth,
                                 edgecolor='none', facecolor=facecolor, alpha=0.1)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=linewidth,
                                 edgecolor=edgecolor, facecolor='none', linestyle='-', alpha=alpha)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=0,
                                 edgecolor=edgecolor, facecolor='none', linestyle='-', hatch=hatch, alpha=0.2)
        ax.add_patch(rect)

    plt.xticks([], [])
    plt.yticks([], [])

    legend_elements = [Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45),
                             label='Table', hatch='//////', alpha=0.3),
                       Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1),
                             label='Table (rotated)', hatch='//////', alpha=0.3)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02), loc='upper center', borderaxespad=0,
               fontsize=10, ncol=2)
    plt.gcf().set_size_inches(10, 10)
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

    return

def visualize_cells(img, cells, out_path):
    plt.imshow(img, interpolation="lanczos")
    plt.gcf().set_size_inches(20, 20)
    ax = plt.gca()

    for cell in cells:
        bbox = cell['bbox']

        if cell['column header']:
            facecolor = (0.9, 0, 0.9)
            edgecolor = (0.9, 0, 0.9)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        elif cell['projected row header']:
            facecolor = (0.95, 0.6, 0.1)
            edgecolor = (0.95, 0.6, 0.1)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        elif cell['row header']:
            facecolor = (1, 0, 0)
            edgecolor = (1, 0, 0)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        elif cell['projected column header']:
            facecolor = (0.49, 0.15, 0.8)
            edgecolor = (0.49, 0.15, 0.8)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        elif cell['table name']:
            facecolor = (0, 0.8, 0)
            edgecolor = (0, 0.8, 0)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        else:
            facecolor = (0.3, 0.74, 0.8)
            edgecolor = (0.3, 0.7, 0.6)
            alpha = 0.3
            linewidth = 2
            hatch = '\\\\\\\\\\\\'

        rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=linewidth,
                                 edgecolor='none', facecolor=facecolor, alpha=0.1)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=linewidth,
                                 edgecolor=edgecolor, facecolor='none', linestyle='-', alpha=alpha)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=0,
                                 edgecolor=edgecolor, facecolor='none', linestyle='-', hatch=hatch, alpha=0.2)
        ax.add_patch(rect)

    plt.xticks([], [])
    plt.yticks([], [])

    legend_elements = [Patch(facecolor=(0.3, 0.74, 0.8), edgecolor=(0.3, 0.7, 0.6),
                             label='Data cell', hatch='\\\\\\\\\\\\', alpha=0.3),
                       Patch(facecolor=(1, 0, 0), edgecolor=(1, 0, 0),
                             label='Row header cell', hatch='//////', alpha=0.3),
                       Patch(facecolor=(0.9, 0, 0.9), edgecolor=(0.9, 0, 0.9),
                             label='Column header cell', hatch='//////', alpha=0.3),
                       Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1),
                             label='Projected row header cell', hatch='//////', alpha=0.3),
                       Patch(facecolor=(0, 0.8, 0), edgecolor=(0, 0.8, 0),
                             label='Table name cell', hatch='//////', alpha=0.3),
                       Patch(facecolor=(0.49, 0.15, 0.8), edgecolor=(0.49, 0.15, 0.8),
                             label='Projected column header cell', hatch='//////', alpha=0.3)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02), loc='upper center', borderaxespad=0,
               fontsize=10, ncol=3)
    plt.gcf().set_size_inches(10, 10)
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

    return

class TableExtractionPipeline(object):
    def __init__(self, det_device=None, str_device=None,
                 det_model=None, str_model=None,
                 det_model_path=None, str_model_path=None,
                 det_config_path=None, str_config_path=None,
                 model_dir=None, in_dir=None, out_dir=None):

        self.det_device = det_device
        self.str_device = str_device
        self.model_dir = model_dir
        self.in_dir = in_dir
        self.out_dir = out_dir

        self.det_class_name2idx = get_class_map('detection')
        self.det_class_idx2name = {v: k for k, v in self.det_class_name2idx.items()}
        self.det_class_thresholds = detection_class_thresholds

        self.str_class_name2idx = get_class_map('structure')
        self.str_class_idx2name = {v: k for k, v in self.str_class_name2idx.items()}
        self.str_class_thresholds = structure_class_thresholds

        if not det_config_path is None:
            with open(det_config_path, 'r') as f:
                det_config = json.load(f)
            det_args = type('Args', (object,), det_config)
            det_args.device = det_device
            self.det_model, _, _ = build_model(det_args)
            print("Detection model initialized.")

            if not det_model_path is None:
                self.det_model.load_state_dict(torch.load(det_model_path,
                                                          map_location=torch.device(det_device)))
                self.det_model.to(det_device)
                self.det_model.eval()
                print("Detection model weights loaded.")
            else:
                self.det_model = None

        if not str_config_path is None:
            with open(str_config_path, 'r') as f:
                str_config = json.load(f)
            str_args = type('Args', (object,), str_config)
            str_args.device = str_device
            self.str_model, _, _ = build_model(str_args)
            print("Structure model initialized.")

            if not str_model_path is None:
                self.str_model.load_state_dict(torch.load(str_model_path,
                                                          map_location=torch.device(str_device)))
                self.str_model.to(str_device)
                self.str_model.eval()
                print("Structure model weights loaded.")
            else:
                self.str_model = None

    def __call__(self, page_image, page_tokens=None):
        return self.extract(self, page_image, page_tokens)

    def detect(self, img, tokens=None, out_objects=True, out_crops=False, crop_padding=10):
        out_formats = {}
        if self.det_model is None:
            print("No detection model loaded.")
            return out_formats

        # Transform the image how the model expects it
        img_tensor = detection_transform(img)

        # Run input image through the model
        outputs = self.det_model([img_tensor.to(self.det_device)])

        # Post-process detected objects, assign class labels
        objects = outputs_to_objects(outputs, img.size, self.det_class_idx2name)
        if out_objects:
            out_formats['objects'] = objects
        if not out_crops:
            return out_formats

        # Crop image and tokens for detected table
        if out_crops:
            tables_crops = objects_to_crops(img, tokens, objects, self.det_class_thresholds,
                                            padding=crop_padding)
            out_formats['crops'] = tables_crops

        return out_formats

    def recognize(self, img, tokens=None, out_objects=False, out_cells=False,
                  out_html=False, out_csv=False):
        out_formats = {}
        if self.str_model is None:
            print("No structure model loaded.")
            return out_formats

        if not (out_objects or out_cells or out_html or out_csv):
            print("No output format specified")
            return out_formats

        # Transform the image how the model expects it
        img = enhance_image(img).convert('RGB')
        img_tensor = structure_transform(img)

        # Run input image through the model
        outputs = self.str_model([img_tensor.to(self.str_device)])

        # Post-process detected objects, assign class labels
        objects = outputs_to_objects(outputs, img.size, self.str_class_idx2name)
        if out_objects:
            out_formats['objects'] = objects
        if not (out_cells or out_html or out_csv):
            return out_formats

        # Further process the detected objects so they correspond to a consistent table
        tables_structure = objects_to_structures(objects, tokens, self.str_class_thresholds)

        # Enumerate all table cells: grid cells and spanning cells
        tables_cells = [structure_to_cells(structure, tokens)[0] for structure in tables_structure]
        if out_cells:
            out_formats['cells'] = tables_cells
        if not (out_html or out_csv):
            return out_formats

        # Convert cells to HTML
        if out_html:
            tables_htmls = [cells_to_html(cells) for cells in tables_cells]
            out_formats['html'] = tables_htmls

        # Convert cells to CSV, including flattening multi-row column headers to a single row
        if out_csv:
            tables_csvs = [cells_to_csv(cells) for cells in tables_cells]
            out_formats['csv'] = tables_csvs
            out_formats['xlsx'] = tables_csvs
            #tables_csvs_df = pd.DataFrame(tables_csvs)
            #writer = pd.ExcelWriter(os.path.join(), engine='xlsxwriter')
            #tables_csvs_df.to_excel(writer, sheet_name='1', index=False)
            #writer.save()

        return out_formats

    def extract(self, img, tokens=None, out_objects=True, out_crops=False, out_cells=False,
                out_html=False, out_csv=False, crop_padding=10):

        detect_out = self.detect(img, tokens=tokens, out_objects=False, out_crops=True,
                                 crop_padding=crop_padding)
        cropped_tables = detect_out['crops']

        extracted_tables = []
        for table in cropped_tables:
            img = table['image']
            tokens = table['tokens']

            extracted_table = self.recognize(img, tokens=tokens, out_objects=out_objects,
                                             out_cells=out_cells, out_html=out_html, out_csv=out_csv)
            extracted_table['image'] = img
            extracted_table['tokens'] = tokens
            extracted_tables.append(extracted_table)

        return extracted_tables

    def perform_final_step(self, excel_folder_path, pdf_folder_path):
        """
        This function will perform the final step as mentioned in the
        documentation. It will extract and structure the values from
        the excel files using regex patterns and return them as a
        dictionary. This will be the major function that will perform
            the final step for all the excel files in the folder
            """

        print("Performing final Step")
        all_files = self.fs_folder(
            path_to_excel_folder=excel_folder_path,
            path_to_pdf_folder=pdf_folder_path
        )
        self.extracted_values = all_files
        print("final Step completed")

    def fs_folder(
            self,
            path_to_excel_folder: str,
            path_to_pdf_folder: str
    ) -> dict:
        """
        This is an internal function that will perform the final step
        for all the excel files in the folder specified.
        """

        list_of_files = get_list_of_files_with_ext(
            path_to_folder=path_to_excel_folder,
            ext=".xlsx",
            verbose=True
        )

        all_files_extracted = {}

        # Go through all the files one by one and get the
        # values that were extracted
        for file in list_of_files:
            #filename = str(basename(file)).rsplit(sep=".")[0]
            filename = os.path.splitext(file)[0]

            # Just run the function for now
            extracted_vals = self.fs_file(
                path_to_excel_file=file,
                path_to_pdf_file=filename + ".pdf"
            )

            all_files_extracted[filename] = extracted_vals

        return all_files_extracted

    def extract_data_with_llm(row_data, field_description):
        # Format your prompt for the LLM
        prompt = f"Extract the {field_description} from the following data: {row_data}"

        # Query GPT-4 (or any other LLM)
        response = openai.Completion.create(
            engine="gpt-4",  # You can also use 'gpt-3.5-turbo'
            prompt=prompt,
            max_tokens=100,
            temperature=0  # Low temperature for deterministic responses
        )

        # Extract the text from the LLM response
        extracted_value = response.choices[0].text.strip()
        return extracted_value

    def save_to_excel(
            self,
            path: str = "extracted_data.xlsx"
    ) -> None:
        """
        This is the path where the excel file containing the extracted
        values will be saved.

        Args:
            path:
                This is the path to the excel file where the extracted
                values will be saved.
        """

        print()

        print("Saving to excel")
        print("---------------")

        final_list = []

        for name, prop_type in self.extracted_values.items():

            # Get the values
            thermal = prop_type.get("thermal")
            electrical = prop_type.get("electrical")
            electrical_nmot = prop_type.get("electrical_nmot")
            pack = prop_type.get("pack")

            year = prop_type.get("misc").get("year")

            length = prop_type.get("mech").get("length")
            width = prop_type.get("mech").get("width")
            height = prop_type.get("mech").get("height")

            # Assuming equal lengths of extracted arrays of values
            if electrical is not None:
                value_count = []

                for value_type, value_list in electrical.items():
                    if value_list is not None:
                        value_count.append(len(value_list))

                # most_freq_count = mode(value_count)
                value_count_cleaned = np.nan_to_num(value_count, nan=np.nan)

                # Filter out NaN values
                value_count_cleaned = value_count_cleaned[~np.isnan(value_count_cleaned)]

                if len(value_count_cleaned) == 0:
                    most_freq_count = np.nan
                else:
                    most_freq_count = int(stats.mode(value_count_cleaned)[0])
            else:
                most_freq_count = 1
            """elif electrical_nmot is not None:
                value_count = []

                for value_type, value_list in electrical_nmot.items():
                    if value_list is not None:
                        value_count.append(len(value_list))

                # most_freq_count = mode(value_count)
                value_count_cleaned = np.nan_to_num(value_count, nan=np.nan)

                # Filter out NaN values
                value_count_cleaned = value_count_cleaned[~np.isnan(value_count_cleaned)]

                if len(value_count_cleaned) == 0:
                    most_freq_count = np.nan
                else:
                    most_freq_count = int(stats.mode(value_count_cleaned)[0])"""


            if math.isnan(most_freq_count):
                curr_file_list = [[]]
            else:
                name = name.split("\\")[-1]
                curr_file_list = [[name] * most_freq_count]
                curr_file_list.append([year] * most_freq_count)

                curr_file_list.append([length] * most_freq_count)
                curr_file_list.append([width] * most_freq_count)
                curr_file_list.append([height] * most_freq_count)

                # Adding electrical properties
                elec_prop_types = ["eff", "pmpp", "vmpp", "impp", "voc", "isc", "ff"]

                for prop in elec_prop_types:

                    if electrical is not None:
                        vals = electrical.get(prop)
                    else:
                        vals = None

                    if vals is None:
                        curr_file_list.append([""] * most_freq_count)
                    else:
                        if len(vals) == most_freq_count:
                            curr_file_list.append(vals)
                        elif len(vals) < most_freq_count:
                            # Append empty strings to make the length equal to most_freq_count
                            curr_file_list.append(vals + [""] * (most_freq_count - len(vals)))
                        elif len(vals) > most_freq_count:
                            # Trim the list to match most_freq_count
                            curr_file_list.append(vals[:most_freq_count])

                # adding electrival propertoes in Nominal module operating temperature
                elec_nmot_prop_types = ["eff_nmot", "pmpp_nmot", "vmpp_nmot", "impp_nmot", "voc_nmot", "ise_nmot"]
                for prop in elec_nmot_prop_types:
                    if electrical_nmot is not None:
                        vals = electrical_nmot.get(prop)
                    else:
                        vals = None

                    if vals is None:
                        curr_file_list.append([""] * most_freq_count)
                    else:
                        if len(vals) == most_freq_count:
                            curr_file_list.append(vals)
                        elif len(vals) < most_freq_count:
                            # Append empty strings to make the length equal to most_freq_count
                            curr_file_list.append(vals + [""] * (most_freq_count - len(vals)))
                        elif len(vals) > most_freq_count:
                            # Trim the list to match most_freq_count
                            curr_file_list.append(vals[:most_freq_count])

                # Adding thermal properties
                thermal_prop_types = ["isc", "pmpp", "voc"]

                for prop in thermal_prop_types:

                    if thermal is not None:
                        vals = thermal.get(prop)
                    else:
                        vals = None

                    if vals is None or len(vals) == 0:
                        curr_file_list.append([""] * most_freq_count)
                    else:
                        curr_file_list.append(vals * most_freq_count)

                        """# Adding thermal properties
                        thermal_prop_types = ["isc", "pmpp", "voc"]

                        for prop in thermal_prop_types:

                            if thermal is not None:
                                vals = thermal.get(prop)
                            else:
                                vals = None

                            if vals is None:
                                curr_file_list.append([""] * most_freq_count)
                            else:
                                curr_file_list.append(vals * most_freq_count)"""

                #Adding packaging properties
                pack_prop_types = ["pcs_pallet", "pallet_containter"]

                for prop in pack_prop_types:
                    if pack is not None:
                        vals = pack.get(prop)
                    else:
                        vals = None

                    if vals is None or len(vals) == 0:
                        curr_file_list.append([""] * most_freq_count)
                    else:
                        curr_file_list.append(vals * most_freq_count)

            # Transpose the list
            curr_file_list = list(map(list, zip(*curr_file_list)))

            final_list.extend(curr_file_list)

        # Create a dataframe
        final_df = pd.DataFrame(final_list,
                                columns=[
                                    "name",
                                    "year",
                                    "length",
                                    "width",
                                    "height",
                                    "E/eff",
                                    "E/pmpp",
                                    "E/vmpp",
                                    "E/impp",
                                    "E/voc",
                                    "E/isc",
                                    "E/ff",
                                    "ENMOT/eff",
                                    "ENMOT/pmpp",
                                    "ENMOT/vmpp",
                                    "ENMOT/impp",
                                    "ENMOT/voc",
                                    "ENMOT/isc",
                                    "T/isc",
                                    "T/pmpp",
                                    "T/voc",
                                    "P/pcs_pallet",
                                    "P/pallet_container"
                                ]
                                )


        final_df.to_excel(path)
        print("final excel created")
        # print(final_df)

        # Write to the excel file
        #with pd.ExcelWriter(path=path, mode='w') as writer:
            #final_df.to_excel(writer,index=False)

        #wb = openpyxl.Workbook()
        #ws = wb.active
        #ws.append(final_df)
        #wb.save(path)

    def fs_file(
            self,
            path_to_excel_file: str,
            path_to_pdf_file: str
    ) -> dict:
        """
        This is an internal function that will perform the final step
        for the excel file specified and return a dictionary of items
        that were extracted. It will combine the electrical and thermal
        properties together into a single dictionary and just keep the
        values that were extracted and not the rows where they were
        found on.
        """

        # Load the yaml file that contains all the patterns for
        # detecting the correct columns and the values
        with open(self.model_dir + "/patterns.yaml", "r", encoding='utf-8') as stream:
            try:
                patterns = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print(e)

        """with open(self.model_dir + "/intent.yaml", "r", encoding='utf-8') as stream:
            try:
                patterns = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print(e)"""


        # Get type specific patterns
        elec_patterns_STC = patterns.get("electrical")
        elec_patterns_NMOT = patterns.get("electrical_nmot")
        therm_patterns = patterns.get("temperature")
        mech_patterns = patterns.get("mechanical")
        pack_patterns = patterns.get("packaging")

        curr_ds = Datasheet(
            path_to_pdf=path_to_pdf_file,
            path_to_excel=path_to_excel_file,
            path_to_clf=self.model_dir + "/nb_classifier_latest.pickle",
            path_to_vec=self.model_dir + "/vectoriser_latest.pickle"
        )

        curr_ds.extract_electrical_props(patterns=elec_patterns_STC)
        elec_extracted = curr_ds.extracted_elec

        curr_ds.extract_electrical_props_nmot(patterns=elec_patterns_NMOT)
        elec_extracted_nmot = curr_ds.extracted_elec_nmot

        curr_ds.extract_temp_props(patterns=therm_patterns)
        therm_extracted = curr_ds.extracted_temp

        curr_ds.extract_pack_props(patterns=pack_patterns)
        pack_extracted = curr_ds.extracted_pack

        curr_ds.extract_mech_props(patterns=mech_patterns)
        mech_extracted = curr_ds.extracted_mech

        curr_ds.extract_misc_props()
        misc_extracted = curr_ds.extracted_misc

        if elec_extracted is not None:
            for key, item in elec_extracted.items():
                vals = item.get("vals")

                if vals is not None:
                    vals = [str(x) for x in vals]

                elec_extracted[key] = vals

        if elec_extracted_nmot is not None:
            for key, item in elec_extracted_nmot.items():
                vals = item.get("vals")

                if vals is not None:
                    vals=[str(x) for x in vals]
                elec_extracted_nmot[key] = vals


        if therm_extracted is not None:
            for key, item in therm_extracted.items():
                vals = item.get("vals")

                if vals is not None:
                    vals = [str(x) for x in vals]

                therm_extracted[key] = vals

        if pack_extracted is not None:
            for key, item in pack_extracted.items():
                vals = item.get("vals")

                if vals is not None:
                    vals = [str(x) for x in vals]

                pack_extracted[key] = vals

        """if mech_extracted is not None:
            for key, item in mech_extracted.items():
                vals = item.get("vals")

                if vals is not None:
                    vals = [str(x) for x in vals]

                mech_extracted[key] = vals"""

        return {
            "electrical": elec_extracted,
            "electrical_nmot": elec_extracted_nmot,
            "thermal": therm_extracted,
            "pack": pack_extracted,
            "mech": mech_extracted,
            "misc": misc_extracted
        }

def output_result(key, val, args, img, img_file):
    if key == 'objects':
        if args.verbose:
            print(val)
        out_file = img_file.replace(".jpg", "_objects.json")
        with open(os.path.join(args.out_dir, out_file), 'w') as f:
            json.dump(val, f)
        if args.visualize:
            out_file = img_file.replace(".jpg", "_fig_tables.jpg")
            out_path = os.path.join(args.out_dir, out_file)
            visualize_detected_tables(img, val, out_path)
    elif not key == 'image' and not key == 'tokens':
        for idx, elem in enumerate(val):
            if key == 'crops':
                for idx, cropped_table in enumerate(val):
                    out_img_file = img_file.replace(".jpg", "_table_{}.jpg".format(idx))

                    # Ensuring that the output directory exists
                    os.makedirs(args.out_dir, exist_ok=True)

                    cropped_table['image'].save(os.path.join(args.out_dir,
                                                             out_img_file), 'JPEG', quality=95, optimize=True)
                    out_words_file = out_img_file.replace(".jpg", "_words.json")
                    with open(os.path.join(args.out_dir, out_words_file), 'w') as f:
                        json.dump(cropped_table['tokens'], f)
            elif key == 'cells':
                out_file = img_file.replace(".jpg", "_{}_objects.json".format(idx))
                with open(os.path.join(args.out_dir, out_file), 'w') as f:
                    json.dump(elem, f)
                if args.verbose:
                    print(elem)
                if args.visualize:
                    out_file = img_file.replace(".jpg", "_fig_cells.jpg")
                    out_path = os.path.join(args.out_dir, out_file)
                    visualize_cells(img, elem, out_path)
            elif key == "csv":
                out_file = img_file.replace(".jpg", "_{}.csv".format(idx))
                with open(os.path.join(args.out_dir, out_file), 'w') as f:
                    if elem is not None:
                        f.write(elem)
                if args.verbose:
                    print(elem)
            elif key == "xlsx":
                # Parse the CSV content
                parsed_data = list(csv.reader(StringIO(elem), delimiter=',', quotechar='"'))

                # Convert the parsed data into a DataFrame
                df = pd.DataFrame(parsed_data)

                out_file =  os.path.join(args.out_dir ,img_file.replace(".jpg", "_{}.xlsx".format(idx)))

                df.to_excel(out_file, index=False, header=False)
                '''out_file = img_file.replace(".jpg", "_{}.xlsx".format(idx))
                wb = openpyxl.Workbook()
                ws = wb.active

                if elem is not None:
                    # Split the string into rows based on newline characters
                    rows = elem.split('\r\n')
                    for row in rows:
                        # Split each row into cells based on comma separation
                        cells = row.split(',')
                        ws.append(cells)
                    wb.save(os.path.join(args.out_dir, out_file))'''




def get_list_of_files_with_ext(
    path_to_folder: str,
    ext: str,
    randomise: bool = False,
    verbose: bool = True
    ) -> list:
    """
    This function will go through all the files in the given
    folder and make a list of files with the provided extension.
    This can be used, for example, to filter out the required
    files in the folder.

    Parameters:
        path_to_folder:
            This is the path to folder that will be scanned
            for the required files.

        ext:
            This is the extension of the files that will be
            selected from the folder.

        randomise:
            If this flag is set to True, then the list of files
            will be shuffled before being returned.

        verbose:
            If this flag is set to True, then this function will
            display the information from the folder.

    Returns:
        list_of_files:
            This is the list of files in the provided
            directory (folder) that matches the extension
            provided. It contains the full path to the files
            not just the name of the files.
    """

    list_of_files = []

    # Evaluate all files in the directory
    for file in listdir(path_to_folder):

        # Skip the hidden files
        # In linux and macOS, the hidden files start
        # with '.'
        if not file.startswith('.'):

            # Get the files with the specified extension
            if file.endswith(ext):
                full_path = join(path_to_folder, file)
                list_of_files.append(full_path)

    if verbose:
        print()
        print("Looking for " + ext + " files in folder: " + path_to_folder)
        print()
        print("Total " + ext + " files found: " + str(len(list_of_files)))

    # Shuffle the list of files captured
    if randomise:
        random.shuffle(list_of_files)

    return list_of_files


def merge_tables(excel_files_folder, output_folder, pdf_files):
    for pdf_file in pdf_files:
        # Get the base name of the PDF file
        base_name = os.path.splitext(pdf_file)[0]

        # Initialize a list to store DataFrames
        dfs = []

        # Iterate through Excel files in the same folder
        for filename in os.listdir(excel_files_folder):
            if filename.startswith(base_name) and filename.endswith('.xlsx'):
                # Read the Excel file
                df = pd.read_excel(os.path.join(excel_files_folder, filename))

                # Append the DataFrame to combined_df
                dfs.append(df)

        # Concatenate all DataFrames in dfs list
        if dfs:
            combined_file_path = os.path.join(output_folder, base_name + '.xlsx')
            with pd.ExcelWriter(combined_file_path) as writer:
                for idx, df in enumerate(dfs):
                    # Write each DataFrame to a separate worksheet
                    sheet_name = f'{base_name}_{idx}'
                    df.to_excel(writer, index=False, header=False, sheet_name=sheet_name)

            '''combined_df = pd.concat(dfs, ignore_index=True)

            # Write combined_df to a new Excel file
            combined_file_path = os.path.join(output_folder, base_name + '.xlsx')
            with pd.ExcelWriter(combined_file_path) as writer:
                combined_df.to_excel(writer, index=False, sheet_name=base_name)'''
        else:
            print("No matching files found for:", pdf_file)

'''def csv_to_excel(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            csv_file = os.path.join(folder_path, filename)

            # Read the CSV file
            df = pd.read_csv(csv_file)
            # Extract the filename without the .csv extension for the sheet name
            sheet_name = os.path.splitext(filename)[0]
            # Create the Excel file with the same name as the CSV (minus .csv)
            excel_file = os.path.splitext(csv_file)[0] + ".xlsx"

            # Write the DataFrame to the Excel file with the extracted sheet name
            df.to_excel(excel_file, sheet_name=sheet_name, index=False)

            print(f"Converted '{filename}' to '{excel_file}'.")'''


def main():
    #setting the tesseract path for OCR
    #pytesseract.pytesseract.tesseract_cmd = r'D:\Users\swa86085\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    global pdf_files_dir, image_dir, pdf_files
    pytesseract.pytesseract.tesseract_cmd = r'S:\23502\2\216_PVM\Aktuell\01_Orga\21631_MAS_TeamModulbewertung\03_Arbeitsordner\Swathi_Thiruvengadam\Tesseract-OCR\tesseract.exe'
    #ssh
    #pytesseract.pytesseract.tesseract_cmd = r'/net/s/23502/2/280_PVM/Aktuell/01_Orga/23131_MAS_TeamModulbewertung/03_Arbeitsordner/Swathi_Thiruvengadam/Tesseract-OCR/tesseract.exe'
    start_time = time.time()

    args = get_args()
    print(args.__dict__)
    print('-' * 100)

    # create an output directory if it does not exists
    if not args.out_dir is None and not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Create inference pipeline
    print("Creating inference pipeline")
    pipe = TableExtractionPipeline(det_device=args.detection_device,
                                   str_device=args.structure_device,
                                   det_config_path=args.detection_config_path,
                                   det_model_path=args.detection_model_path,
                                   str_config_path=args.structure_config_path,
                                   str_model_path=args.structure_model_path,
                                   model_dir = args.model_dir,
                                   in_dir = args.in_dir,
                                   out_dir = args.out_dir)

    # Load images
    # convert the pdf file to images
    if args.mode == 'detect' or args.mode == 'extract':
        pdf_files = [f for f in os.listdir(args.in_dir) if f.lower().endswith(('.pdf'))]

        # Directory for storing intermediate images
        image_dir = os.path.join(args.in_dir, '..', 'images')
        os.makedirs(image_dir, exist_ok=True)

        for pdf in pdf_files:
            pdf_path = os.path.join(args.in_dir, pdf)
            pdf_image = convert_from_path(pdf_path, dpi=300)
            image_base_path = os.path.join(image_dir, os.path.splitext(pdf)[0])
            #for idx in range(len(pdf_image)):
                #pdf_image[idx].save(image_path + '_' + str(idx + 1) + '.jpg', 'JPEG')
            for idx, image in enumerate(pdf_image):
                #enhanced_image = enhance_image(image)
                image_path = f"{image_base_path}_{idx + 1}.jpg"
                image.save(image_path, 'JPEG')

    elif args.mode == 'recognize':
        pdf_files_dir = os.path.join(args.in_dir, '..', 'pdf')
        pdf_files = [f for f in os.listdir(pdf_files_dir) if f.lower().endswith(('.pdf'))]
        image_dir = os.path.join(args.in_dir, '..', 'images')
        os.makedirs(image_dir, exist_ok=True)

        for pdf in pdf_files:
            pdf_path = os.path.join(pdf_files_dir, pdf)
            pdf_image = convert_from_path(pdf_path, dpi=300)
            image_path = os.path.join(image_dir, os.path.splitext(pdf)[0])
              #for idx in range(len(pdf_image)):
                #pdf_image[idx].save(image_path + '_' + str(idx + 1) + '.jpg', 'JPEG')
            for idx, image in enumerate(pdf_image):
                #enhanced_image = enhance_image(image)
                image_path = f"{image_path}_{idx + 1}.jpg"
                image.save(image_path, 'JPEG')


    # create the OCR words
    if args.mode == 'detect' or args.mode == 'extract':
        words_dir = "../inferences/detectionwords"
        # OCR program to extract the words for all the image files present in our folder
        extract_words_from_images(image_dir, words_dir)
    elif args.mode == 'recognize':
        words_dir = "../inferences/words"
        image_dir = args.in_dir


    # List only image files in the input folder
    img_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    num_files = len(img_files)
    random.shuffle(img_files)

    # note : change directory paths to make more sense
    img_input_folder = os.path.join(image_dir, '..', 'detectionOutput')
    words_output_folder = os.path.join(image_dir, '..', 'words')

    for count, img_file in enumerate(img_files):
        print("({}/{})".format(count + 1, num_files))
        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path).convert('RGB')

        #latest
        #img_path = image_dir +'/' +  img_file
        #img = Image.open(img_path).convert('RGB')
        print("Image loaded.")

        if not words_dir is None:
            tokens_path = words_dir + "/" + img_file.replace(".jpg", "_words.json")
            # tokens_path = os.path.join(args.words_dir, img_file.replace(".jpg", ".json"))
            with open(tokens_path, 'r') as f:
                tokens = json.load(f)

                # Handle dictionary format
                if type(tokens) is dict and 'words' in tokens:
                    tokens = tokens['words']

                # 'tokens' is a list of tokens
                # Need to be in a relative reading order
                # If no order is provided, use current order
                for idx, token in enumerate(tokens):
                    if not 'span_num' in token:
                        token['span_num'] = idx
                    if not 'line_num' in token:
                        token['line_num'] = 0
                    if not 'block_num' in token:
                        token['block_num'] = 0
        else:
            tokens = []

        if args.mode == 'recognize':
            extracted_table = pipe.recognize(img, tokens, out_objects=args.objects, out_cells=args.csv,
                                             out_html=args.html, out_csv=args.csv)
            print("Table(s) recognized.")

            for key, val in extracted_table.items():
                output_result(key, val, args, img, img_file)

            # merge into same file
            merge_tables(args.out_dir, pdf_files_dir, pdf_files)


        if args.mode == 'detect':
            detected_tables = pipe.detect(img, tokens, out_objects=args.objects, out_crops=args.crops)
            print("Table(s) detected.")

            for key, val in detected_tables.items():
                output_result(key, val, args, img, img_file)

        if args.mode == 'extract':
            extracted_tables = pipe.extract(img, tokens, out_objects=args.objects, out_cells=args.csv,
                                            out_html=args.html, out_csv=args.csv,
                                            crop_padding=args.crop_padding)
            print("Table(s) extracted.")

            for table_idx, extracted_table in enumerate(extracted_tables):
                for key, val in extracted_table.items():
                    output_result(key, val, args, extracted_table['image'],
                                  img_file.replace('.jpg', '_{}.jpg'.format(table_idx)))

            # merge into same file
            merge_tables(args.out_dir, args.in_dir, pdf_files)

    if args.mode == 'detect':
        extract_words_from_images(img_input_folder, words_output_folder)

    if args.mode == 'recognize':
        pipe.perform_final_step(pdf_files_dir, pdf_files_dir)
        pipe.save_to_excel(path=os.path.join(pdf_files_dir, "extracted_tables.xlsx"))

    if args.mode == 'extract':
        pipe.perform_final_step(args.in_dir, args.in_dir)
        pipe.save_to_excel(path=os.path.join(args.in_dir, "extracted_tables.xlsx"))

    end_time = time.time()

    total_time = end_time - start_time
    print(f"Total time taken by Table-Transformer to extract tabular data is : {total_time} seconds")

if __name__ == "__main__":
    main()