import matplotlib.pyplot as plt
import numpy as np
from labelImg import labelImg

# Data
labels = ['Solar cells', 'Solar Module']

precision_LT = [0.989, 0.50]
recall_LT = [0.940, 0.50]
f1_LT = [0.964, 0.50]

precision_TATR = [0.6739, 0.5529]
recall_TATR = [0.6739, 0.5529]
f1_TATR = [0.6739, 0.5529]

# Graph 1: Grouped Bar Chart for Precision, Recall, and F1 Score
fig, ax = plt.subplots()
bar_width = 0.2
index = np.arange(len(labels))

bar1 = ax.bar(index - bar_width, precision_LT, bar_width, label='Precision (LT)')
bar2 = ax.bar(index, recall_LT, bar_width, label='Recall (LT)')
bar3 = ax.bar(index + bar_width, f1_LT, bar_width, label='F1 Score (LT)')

bar4 = ax.bar(index + 2*bar_width, precision_TATR, bar_width, label='Precision (TATR)')
bar5 = ax.bar(index + 3*bar_width, recall_TATR, bar_width, label='Recall (TATR)')
bar6 = ax.bar(index + 4*bar_width, f1_TATR, bar_width, label='F1 Score (TATR)')

ax.set_xlabel('Datasheets')
ax.set_ylabel('Scores')
ax.set_title('Precision, Recall, and F1 Score Comparison')
ax.set_xticks(index + 1.5*bar_width)
ax.set_xticklabels(labels)
ax.legend()
plt.show()

# Graph 2: Grouped Bar Chart for Precision, Recall, and F1 Score
fig, ax = plt.subplots()
bar_width = 0.35

bar1 = ax.bar(index - bar_width/2, precision_LT, bar_width, label='Precision (LT)')
bar2 = ax.bar(index - bar_width/2, recall_LT, bar_width, label='Recall (LT)')
bar3 = ax.bar(index - bar_width/2, f1_LT, bar_width, label='F1 Score (LT)')

bar4 = ax.bar(index + bar_width/2, precision_TATR, bar_width, label='Precision (TATR)')
bar5 = ax.bar(index + bar_width/2, recall_TATR, bar_width, label='Recall (TATR)')
bar6 = ax.bar(index + bar_width/2, f1_TATR, bar_width, label='F1 Score (TATR)')

ax.set_xlabel('Datasheets')
ax.set_ylabel('Scores')
ax.set_title('Precision, Recall, and F1 Score Comparison')
ax.set_xticks(index)
ax.set_xticklabels(labels)
ax.legend()
plt.show()

# Graph 3: Precision-Recall Curve
from sklearn.metrics import precision_recall_curve

precision_LT, recall_LT, _ = precision_recall_curve([1, 0], [1, 0], pos_label=1)
precision_TATR, recall_TATR, _ = precision_recall_curve([1, 0], [1, 0], pos_label=1)

fig, ax = plt.subplots()
ax.plot(recall_LT, precision_LT, label='LT')
ax.plot(recall_TATR, precision_TATR, label='TATR')

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend()
plt.show()

# Graph 4: F1 Score Trend Line
fig, ax = plt.subplots()
ax.plot(labels, f1_LT, label='LT', marker='o')
ax.plot(labels, f1_TATR, label='TATR', marker='o')

ax.set_xlabel('Datasheets')
ax.set_ylabel('F1 Score')
ax.set_title('F1 Score Trend Line')
ax.legend()
plt.show()

# Graph 5: Confusion Matrix Heatmap
conf_matrix_LT = np.array([[0.989, 0.011], [0.06, 0.50]])
conf_matrix_TATR = np.array([[0.6739, 0.3261], [0.4471, 0.5529]])

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

im1 = ax[0].imshow(conf_matrix_LT, cmap='Blues', vmin=0, vmax=1)
ax[0].set_title('LT Confusion Matrix')
ax[0].set_xticks([0, 1])
ax[0].set_yticks([0, 1])
ax[0].set_xticklabels(['Predicted 0', 'Predicted 1'])
ax[0].set_yticklabels(['Actual 0', 'Actual 1'])
plt.colorbar(im1, ax=ax[0])

im2 = ax[1].imshow(conf_matrix_TATR, cmap='Blues', vmin=0, vmax=1)
ax[1].set_title('TATR Confusion Matrix')
ax[1].set_xticks([0, 1])
ax[1].set_yticks([0, 1])
ax[1].set_xticklabels(['Predicted 0', 'Predicted 1'])
ax[1].set_yticklabels(['Actual 0', 'Actual 1'])
plt.colorbar(im2, ax=ax[1])

plt.show()










"""import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['Solar cells', 'Solar Module']

precision_LT = [0.989, 0.50]
recall_LT = [0.940, 0.50]
f1_LT = [0.964, 0.50]

precision_TATR = [0.6739, 0.5529]
recall_TATR = [0.6739, 0.5529]
f1_TATR = [0.6739, 0.5529]

# Graph 1: Grouped Bar Chart for Precision, Recall, and F1 Score
fig, ax = plt.subplots()
bar_width = 0.2
index = np.arange(len(labels))

bar1 = ax.bar(index - bar_width, precision_LT, bar_width, label='Precision (LT)')
bar2 = ax.bar(index, recall_LT, bar_width, label='Recall (LT)')
bar3 = ax.bar(index + bar_width, f1_LT, bar_width, label='F1 Score (LT)')

bar4 = ax.bar(index + 2*bar_width, precision_TATR, bar_width, label='Precision (TATR)')
bar5 = ax.bar(index + 3*bar_width, recall_TATR, bar_width, label='Recall (TATR)')
bar6 = ax.bar(index + 4*bar_width, f1_TATR, bar_width, label='F1 Score (TATR)')

ax.set_xlabel('Datasheets')
ax.set_ylabel('Scores')
ax.set_title('Precision, Recall, and F1 Score Comparison')
ax.set_xticks(index + 1.5*bar_width)
ax.set_xticklabels(labels)
ax.legend()
plt.show()

# Graph 2: Grouped Bar Chart for Precision, Recall, and F1 Score
fig, ax = plt.subplots()
bar_width = 0.35

bar1 = ax.bar(index - bar_width/2, precision_LT, bar_width, label='Precision (LT)')
bar2 = ax.bar(index - bar_width/2, recall_LT, bar_width, label='Recall (LT)')
bar3 = ax.bar(index - bar_width/2, f1_LT, bar_width, label='F1 Score (LT)')

bar4 = ax.bar(index + bar_width/2, precision_TATR, bar_width, label='Precision (TATR)')
bar5 = ax.bar(index + bar_width/2, recall_TATR, bar_width, label='Recall (TATR)')
bar6 = ax.bar(index + bar_width/2, f1_TATR, bar_width, label='F1 Score (TATR)')

ax.set_xlabel('Datasheets')
ax.set_ylabel('Scores')
ax.set_title('Precision, Recall, and F1 Score Comparison')
ax.set_xticks(index)
ax.set_xticklabels(labels)
ax.legend()
plt.show()

# Graph 3: Precision-Recall Curve
from sklearn.metrics import precision_recall_curve

precision_LT, recall_LT, _ = precision_recall_curve([1, 0], [1, 0], pos_label=1)
precision_TATR, recall_TATR, _ = precision_recall_curve([1, 0], [1, 0], pos_label=1)

fig, ax = plt.subplots()
ax.plot(recall_LT, precision_LT, label='LT')
ax.plot(recall_TATR, precision_TATR, label='TATR')

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend()
plt.show()

# Graph 4: F1 Score Trend Line
fig, ax = plt.subplots()
ax.plot(labels, f1_LT, label='LT', marker='o')
ax.plot(labels, f1_TATR, label='TATR', marker='o')

ax.set_xlabel('Datasheets')
ax.set_ylabel('F1 Score')
ax.set_title('F1 Score Trend Line')
ax.legend()
plt.show()

# Graph 5: Confusion Matrix Heatmap
conf_matrix_LT = np.array([[0.989, 0.011], [0.06, 0.50]])
conf_matrix_TATR = np.array([[0.6739, 0.3261], [0.4471, 0.5529]])

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

im1 = ax[0].imshow(conf_matrix_LT, cmap='Blues', vmin=0, vmax=1)
ax[0].set_title('LT Confusion Matrix')
ax[0].set_xticks([0, 1])
ax[0].set_yticks([0, 1])
ax[0].set_xticklabels(['Predicted 0', 'Predicted 1'])
ax[0].set_yticklabels(['Actual 0', 'Actual 1'])
plt.colorbar(im1, ax=ax[0])

im2 = ax[1].imshow(conf_matrix_TATR, cmap='Blues', vmin=0, vmax=1)
ax[1].set_title('TATR Confusion Matrix')
ax[1].set_xticks([0, 1])
ax[1].set_yticks([0, 1])
ax[1].set_xticklabels(['Predicted 0', 'Predicted 1'])
ax[1].set_yticklabels(['Actual 0', 'Actual 1'])
plt.colorbar(im2, ax=ax[1])

plt.show()"""













"""import os
import csv
import yaml
from sklearn.metrics import precision_score, recall_score
from Levenshtein import distance

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
    #Flatten a dictionary with alternating keys and values.
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
            predicted_list = [item.replace('|', '') for item in predicted_list]

            if predicted_list and distance(str(key), str(predicted_list[0])) <= 3:
                # Flatten the matching list from predicted_data
                gd_length = 1 if isinstance(ground_truth[key], str) else len(ground_truth[key])
                pred_length = len(predicted_list)

                if(pred_length == (gd_length+1)):
                    predicted_data_flat.extend(predicted_list)
                elif(pred_length < (gd_length+1)):
                    padding = (gd_length+1) - pred_length
                    predicted_data_flat.extend(predicted_list)
                    predicted_data_flat.extend([''] * padding)
                #values = [str(item).replace('|', '').strip() for item in ground_truth[key]]
                #values = [str(item).replace('|', '').strip() if isinstance(item, (list, tuple)) else str(item).replace('|','').strip() for item in ground_truth[key]]
                #predicted_data_flat.append(key)
                #predicted_data_flat.extend(values)
                found_matching_list = True
                # Break after finding the first matching list
                break

        # If no matching list is found, append the predicted_data_flat with n elements
        if not found_matching_list:
            n = 1 if isinstance(ground_truth[key], str) else len(ground_truth[key])
            predicted_data_flat.extend([''] * (n + 1))

    # Add a conditional step to pad predicted_data_flat if needed
    if len(ground_truth_flat) > len(predicted_data_flat):
        padding_size = len(ground_truth_flat) - len(predicted_data_flat)
        predicted_data_flat.extend([''] * padding_size)
    elif len(ground_truth_flat) < len(predicted_data_flat):
        padding_size = len(predicted_data_flat) - len(ground_truth_flat)
        ground_truth_flat.extend([''] * padding_size)

    '''print(ground_truth_flat)
    print(predicted_data_flat)
    print(predicted_data)'''
    # Compute precision and recall
    precision = precision_score(ground_truth_flat, predicted_data_flat, average='micro')
    recall = recall_score(ground_truth_flat, predicted_data_flat, average='micro')

    return precision, recall

def flatten_list(lst):
    return [item for sublist in lst for item in sublist]

def evaluate_folder(folder_path, ground_truth_folder):
    precision_sum = 0
    recall_sum = 0
    num_files = 0

    for csv_file in os.listdir(folder_path):
        if csv_file.endswith(".csv"):
            csv_path = folder_path + '/' + csv_file
            yaml_file = ground_truth_folder + '/' + os.path.splitext(csv_file)[0] + ".yml"

            if os.path.exists(yaml_file):
                ground_truth = read_ground_truth(yaml_file)
                predicted_data = read_predicted_data(csv_path)

                precision, recall = evaluate_table_extraction(ground_truth, predicted_data)
                print(f'File: {csv_file}, Precision: {precision:.4f}, Recall: {recall:.4f}')

                precision_sum += precision
                recall_sum += recall
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
"""

