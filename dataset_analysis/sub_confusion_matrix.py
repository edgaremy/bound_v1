import numpy as np

# mother_classes[i]=j means that the class i is a subclass of the mother-class j
def merge_confusion_matrix(confusion_matrix, mother_classes):
    num_classes = len(mother_classes)
    num_mother_classes = max(mother_classes) + 1

    merged_matrix = np.zeros((num_mother_classes, num_mother_classes), dtype=int)

    for i in range(num_classes):
        for j in range(num_classes):
            mother_class_i = mother_classes[i]
            mother_class_j = mother_classes[j]
            merged_matrix[mother_class_i,mother_class_j] += confusion_matrix[i,j]

    return merged_matrix

# Example Usage:
# confusion_matrix = np.array([[13, 1, 2, 0],
#                              [0, 10, 0, 0],
#                              [0, 0, 8, 0],
#                              [0, 0, 0, 9]])
# mother_classes = [0, 1, 0, 1]
# merged_matrix = merge_confusion_matrix(confusion_matrix, mother_classes)
# print(merged_matrix)

def confusion_matrix_stats(conf_mat):
    # Counting true positives, false positives, true negatives, and false negatives
    true_positives = np.diag(conf_mat)
    false_positives = np.sum(conf_mat, axis=1) - true_positives
    false_negatives = np.sum(conf_mat, axis=0) - true_positives
    true_negatives = np.sum(conf_mat) - true_positives - false_positives - false_negatives
    # F1 Score
    f1_score = np.where(true_positives > 0, 2 * true_positives / (2 * true_positives + false_positives + false_negatives), 0)
    return f1_score, true_positives, false_positives, false_negatives, true_negatives