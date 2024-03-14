import numpy as np

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
confusion_matrix = np.array([[13, 1, 2, 0],
                             [0, 10, 0, 0],
                             [0, 0, 8, 0],
                             [0, 0, 0, 9]])
mother_classes = [0, 1, 0, 1]
merged_matrix = merge_confusion_matrix(confusion_matrix, mother_classes)
print(merged_matrix)