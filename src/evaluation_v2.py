import os
import csv
import yaml
from tabulate import tabulate
from sklearn.metrics import precision_score, recall_score

def read_ground_truth(file_path):
    with open(file_path, 'r') as yaml_file:
        ground_truth = yaml.safe_load(yaml_file)
    return ground_truth

def read_predicted_data(file_path):
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        predicted_data = [row for row in reader]
    return predicted_data

def flatten_nested_list(nested_list):
    """Flatten a nested list."""
    return [item for sublist in nested_list for item in (sublist if isinstance(sublist, list) else [sublist])]

def flatten_dict_with_lists(input_dict):
    """Flatten a dictionary with alternating keys and values."""
    flattened_list = []
    for key, value in input_dict.items():
        flattened_list.append(key)
        if isinstance(value, list):
            flattened_list.extend(value)
        else:
            flattened_list.append(value)
    return flattened_list

def evaluate_table_extraction(ground_truth, predicted_data):
    # Flatten ground truth dictionary into a list of interleaved keys and values
    ground_truth_flat = flatten_dict_with_lists(ground_truth)

    # Initialize an empty list to store the corresponding predicted data
    predicted_data_flat = []

    # Inside the loop that iterates over keys in ground_truth
    for key in ground_truth.keys():
        found_matching_list = False
        for predicted_list in predicted_data:
            if predicted_list and str(key) == predicted_list[0]:
                # Flatten the matching list from predicted_data
                #predicted_data_flat.extend(
                    #[item.replace('|', '').strip() for item in flatten_nested_list(predicted_list)])
                values = [str(item).replace('|', '').strip() for item in ground_truth[key]]
                predicted_data_flat.append(key)
                predicted_data_flat.extend(values)
                found_matching_list = True
                break  # Break after finding the first matching list

        # If no matching list is found, append the predicted_data_flat with n elements
        '''if not found_matching_list:
            n = len(ground_truth[key])
            predicted_data_flat.extend([''] * (n+1))'''
        if not found_matching_list:
            if isinstance(ground_truth[key], list):
                # Check if the value is a list before attempting to get its length
                n = len(ground_truth[key])
            else:
                # If it's not a list, treat it as a single element
                n = 1
            predicted_data_flat.extend([''] * (n + 1))
    # Iterate over the keys in ground_truth and find the matching list in predicted_data
    '''for key in ground_truth.keys():
        for predicted_list in predicted_data:
            if predicted_list:
                if key == predicted_list[0]:
                    # Flatten the matching list from predicted_data
                    predicted_data_flat.extend(
                        [item.replace('|', '').strip() for item in flatten_nested_list(predicted_list)])
                    break  # Break after finding the first matching list
                else:'''


    # Add a conditional step to pad predicted_data_flat if needed
    if len(ground_truth_flat) > len(predicted_data_flat):
        padding_size = len(ground_truth_flat) - len(predicted_data_flat)
        predicted_data_flat.extend([''] * padding_size)
    elif len(ground_truth_flat) < len(predicted_data_flat):
        padding_size = len(predicted_data_flat) - len(ground_truth_flat)
        ground_truth_flat.extend([''] * padding_size)

    # Compute precision and recall
    precision = precision_score(ground_truth_flat, predicted_data_flat, average='micro')
    recall = recall_score(ground_truth_flat, predicted_data_flat, average='micro')

    return precision, recall


def flatten_list(lst):
    return [item for sublist in lst for item in sublist]

def compute_classification_metrics(ground_truth, predictions):
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for key in ground_truth.keys():
        for predicted_list in predictions:
            # Check if the predicted list is not empty and the key matches the first element
            if predicted_list and key == predicted_list[0]:
                # Flatten the lists for comparison
                ground_truth_flat = flatten_list([key] + ground_truth[key])
                predictions_flat = flatten_list(predicted_list)

                # Ensure the lists are of the same length by appending None if needed
                max_len = max(len(ground_truth_flat), len(predictions_flat))
                ground_truth_flat += [None] * (max_len - len(ground_truth_flat))
                predictions_flat += [None] * (max_len - len(predictions_flat))

                # Check each element in the flattened lists
                for i in range(len(ground_truth_flat)):
                    if ground_truth_flat[i] == predictions_flat[i]:
                        true_positive += 1
                    elif ground_truth_flat[i] is not None:
                        false_negative += 1  # Ground truth is positive, but prediction is negative
                        false_positive += 1  # Prediction is positive, but ground truth is negative

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1_score

def evaluate_folder(folder_path, ground_truth_folder):
    precision_sum = 0
    recall_sum = 0
    num_files = 0

    for csv_file in os.listdir(folder_path):
        if csv_file.endswith(".csv"):
            #csv_path = os.path.join(folder_path, csv_file)
            csv_path = folder_path + '/' + csv_file
            yaml_file = ground_truth_folder + '/' + os.path.splitext(csv_file)[0] + ".yml"
            #yaml_file = os.path.join(folder_path, os.path.splitext(csv_file)[0] + ".yaml")

            if os.path.exists(yaml_file):
                ground_truth = read_ground_truth(yaml_file)
                predicted_data = read_predicted_data(csv_path)

                #precision, recall = evaluate_table_extraction(ground_truth, predicted_data)

                #print(f'File: {csv_file}, Precision: {precision:.4f}, Recall: {recall:.4f}')
                precision, recall= evaluate_table_extraction(ground_truth, predicted_data)
                print(f'File: {csv_file}, Precision: {precision:.4f}, Recall: {recall:.4f}')

                precision_sum += precision
                recall_sum += recall
                #num_files += 1
                num_files += 1

    print(num_files)

    if num_files > 0:
        average_precision = precision_sum / num_files
        average_recall = recall_sum / num_files
        print(f'Average Precision: {average_precision:.4f}')
        print(f'Average Recall: {average_recall:.4f}')

def main():
    # Replace with your folder paths
    predicted_folder = '../inferences/extractionOutput'
    ground_truth_folder = '../inferences/groundTruth'

    print("Evaluation results for each file:")
    evaluate_folder(predicted_folder, ground_truth_folder)

if __name__ == "__main__":
    main()
