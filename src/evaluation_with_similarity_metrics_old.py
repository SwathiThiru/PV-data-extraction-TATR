import os
import csv
import yaml
from sklearn.metrics import precision_score, recall_score, f1_score
from Levenshtein import distance
import matplotlib.pyplot as plt

def read_ground_truth(file_path):
    with open(file_path, 'r', errors='ignore') as yaml_file:
        ground_truth = yaml.safe_load(yaml_file)
    return ground_truth

def read_predicted_data(file_path):
    with open(file_path, 'r', errors='ignore') as csv_file:
        reader = csv.reader(csv_file)
        predicted_data = [row for row in reader]
    return predicted_data

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
    ground_truth_flat = flatten_dict_with_lists(ground_truth)
    predicted_data_flat = []

    for key in ground_truth.keys():
        found_matching_list = False
        for predicted_list in predicted_data:
            if predicted_list and distance(str(key), str(predicted_list[0])) <= 3:
                predicted_data_flat.append(key)
                values = [str(item).replace('|', '').strip() if isinstance(item, (list, tuple)) else str(item).replace('|','').strip() for item in ground_truth[key]]
                predicted_data_flat.extend(predicted_list[1:len(values)+1])
                found_matching_list = True
                break
        if not found_matching_list:
            n = len(ground_truth[key])
            predicted_data_flat.extend([''] * (n + 1))

    if len(ground_truth_flat) > len(predicted_data_flat):
        padding_size = len(ground_truth_flat) - len(predicted_data_flat)
        predicted_data_flat.extend([''] * padding_size)
    elif len(ground_truth_flat) < len(predicted_data_flat):
        padding_size = len(predicted_data_flat) - len(ground_truth_flat)
        ground_truth_flat.extend([''] * padding_size)

    # Handle NaNs or empty strings
    ground_truth_flat = [str(item) for item in ground_truth_flat]
    predicted_data_flat = [str(item) for item in predicted_data_flat]
    print("Ground Truth Flattened:", ground_truth_flat)
    print("Predicted Data Flattened:", predicted_data_flat)

    precision = precision_score(ground_truth_flat, predicted_data_flat, average='weighted', zero_division=0)
    recall = recall_score(ground_truth_flat, predicted_data_flat, average='micro', zero_division=0)
    f1 = f1_score(ground_truth_flat, predicted_data_flat, average='micro', zero_division=0)

    return precision, recall, f1

def plot_metrics(file_names, precision_list, recall_list, f1_list):
    plt.figure(figsize=(10, 6))
    x = range(len(file_names))
    plt.plot(x, precision_list, label='Precision', marker='o')
    plt.plot(x, recall_list, label='Recall', marker='s')
    plt.plot(x, f1_list, label='F1 Score', marker='^')
    plt.xticks(x, file_names, rotation=45, ha='right')
    plt.xlabel('Files')
    plt.ylabel('Scores')
    plt.title('Precision, Recall, and F1 Score for Each File')
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_folder(folder_path, ground_truth_folder):
    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    num_files = 0

    precision_list = []
    recall_list = []
    f1_list = []
    file_names = []

    for csv_file in os.listdir(folder_path):
        if csv_file.endswith(".csv"):
            csv_path = os.path.join(folder_path, csv_file)
            yaml_file = os.path.join(ground_truth_folder, os.path.splitext(csv_file)[0] + ".yml")

            if os.path.exists(yaml_file):
                ground_truth = read_ground_truth(yaml_file)
                predicted_data = read_predicted_data(csv_path)

                precision, recall, f1 = evaluate_table_extraction(ground_truth, predicted_data)
                print(f'File: {csv_file}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

                precision_sum += precision
                recall_sum += recall
                f1_sum += f1
                if precision != 0 and recall !=0 and f1 != 0:
                    num_files += 1

                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
                file_names.append(csv_file)

    print('Number of files evaluated:', num_files)

    if num_files > 0:
        average_precision = precision_sum / num_files
        average_recall = recall_sum / num_files
        average_f1 = f1_sum / num_files
        print(f'Average Precision: {average_precision:.4f}')
        print(f'Average Recall: {average_recall:.4f}')
        print(f'Average F1 Score: {average_f1:.4f}')

    plot_metrics(file_names, precision_list, recall_list, f1_list)

def main():
    predicted_folder = '../inferences/extractionOutput'
    ground_truth_folder = '../inferences/groundTruth'

    print("Evaluation results for each file:")
    evaluate_folder(predicted_folder, ground_truth_folder)

if __name__ == "__main__":
    main()


"""
import os
import csv
import yaml
from sklearn.metrics import precision_score, recall_score, f1_score
from Levenshtein import distance
import matplotlib.pyplot as plt

def read_ground_truth(file_path):
    with open(file_path, 'r') as yaml_file:
        ground_truth = yaml.safe_load(yaml_file)
    return ground_truth

def read_predicted_data(file_path):
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        predicted_data = [row for row in reader]
    return predicted_data

def flatten_dict_with_lists(input_dict):
    flattened_list = []
    for key, value in input_dict.items():
        flattened_list.append(key)
        if isinstance(value, list):
            flattened_list.extend(value)
        else:
            # Convert non-list values to a list with a single element
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
            if predicted_list and distance(str(key), str(predicted_list[0])) <= 2:
                # Flatten the matching list from predicted_data
                #values = [str(item).replace('|', '').strip() for item in ground_truth[key]]
                values = [str(item).replace('|', '').strip() if isinstance(item, (list, tuple)) else str(item).replace('|','').strip() for item in ground_truth[key]]
                predicted_data_flat.append(key)
                predicted_data_flat.extend(values)
                found_matching_list = True
                # Break after finding the first matching list
                break

        # If no matching list is found, append the predicted_data_flat with n elements
        if not found_matching_list:
            n = len(ground_truth[key])
            predicted_data_flat.extend([''] * (n + 1))

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
    f1 = f1_score(ground_truth_flat, predicted_data_flat, average='micro')

    return precision, recall, f1

def flatten_list(lst):
    return [item for sublist in lst for item in sublist]

def evaluate_folder(folder_path, ground_truth_folder):
    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    num_files = 0

    precision_list = []
    recall_list = []
    f1_list = []

    for csv_file in os.listdir(folder_path):
        if csv_file.endswith(".csv"):
            csv_path = folder_path + '/' + csv_file
            yaml_file = ground_truth_folder + '/' + os.path.splitext(csv_file)[0] + ".yml"

            if os.path.exists(yaml_file):
                ground_truth = read_ground_truth(yaml_file)
                predicted_data = read_predicted_data(csv_path)

                precision, recall, f1  = evaluate_table_extraction(ground_truth, predicted_data)
                print(f'File: {csv_file}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

                precision_sum += precision
                recall_sum += recall
                f1_sum += f1
                num_files += 1

                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)

    print('Number of files evaluated : ',num_files)

    if num_files > 0:
        average_precision = precision_sum / num_files
        average_recall = recall_sum / num_files
        average_f1 = f1_sum / num_files
        print(f'Average Precision: {average_precision:.4f}')
        print(f'Average Recall: {average_recall:.4f}')
        print(f'Average F1 Score: {average_f1:.4f}')



def main():
    # Replace with your folder paths
    predicted_folder = '../inferences/extractionOutput'
    ground_truth_folder = '../inferences/groundTruth'

    print("Evaluation results for each file:")
    evaluate_folder(predicted_folder, ground_truth_folder)

if __name__ == "__main__":
    main()"""