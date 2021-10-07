# Confusion matrix is an nxn matrix. Columns = PREDICTIONS, Rows = REALITY

def get_classes_iou(confusion_matrix, classes):

    classes_iou = []

    correct, wrong_r = _get_correct_and_wrong_pixels_rows(confusion_matrix)
    _, wrong_c = _get_correct_and_wrong_pixels_columns(confusion_matrix)

    for i in range(len(correct)):

        total_pixels = correct[i] + wrong_r[i] + wrong_c[i]

        if i in classes and total_pixels > 0:
            classes_iou.append(correct[i]/total_pixels)
        else:
            classes_iou.append(None)

    return classes_iou

def get_classes_accuracy(confusion_matrix, classes):

    classes_acc = []

    correct, wrong = _get_correct_and_wrong_pixels_columns(confusion_matrix)

    for i in range(len(correct)):

        total_pixels = correct[i] + wrong[i]

        if i in classes and total_pixels > 0:
            classes_acc.append(correct[i] / total_pixels)
        else:
            classes_acc.append(None)

    return classes_acc

def get_mean_pixel_accuracy(confusion_matrix, classes):

    correct, wrong = _get_correct_and_wrong_pixels_columns(confusion_matrix)

    total_correct = 0
    total_wrong = 0

    for i in range(len(correct)):
        if i in classes:
            total_correct += correct[i]
            total_wrong += wrong[i]

    return total_correct/(total_correct + total_wrong)



def get_mean_class_accuracy(confusion_matrix, classes):

    classes_acc = get_classes_accuracy(confusion_matrix, classes)
    classes_acc = [c for c in classes_acc if c is not None]

    if len(classes_acc) == 0:
        return 0.0

    return sum(classes_acc)/len(classes_acc)


def get_mean_iou(confusion_matrix, classes):

    classes_iou = get_classes_iou(confusion_matrix, classes)
    classes_iou = [c for c in classes_iou if c is not None]
    return sum(classes_iou)/len(classes_iou)

def get_mean_accuracy_old_and_new(confusion_matrix, old_classes, new_classes):
    return get_mean_class_accuracy(confusion_matrix, old_classes), get_mean_class_accuracy(confusion_matrix, new_classes)

def get_mean_iou_old_and_new(confusion_matrix, old_classes, new_classes):

    return get_mean_iou(confusion_matrix, old_classes), get_mean_iou(confusion_matrix, new_classes)

# correct = TRUE POSITIVES
# wrong = FALSE NEGATIVES
def _get_correct_and_wrong_pixels_rows(confusion_matrix):
    correct = []
    wrong = []

    n_rows = confusion_matrix.shape[0]
    # for every row

    for r in range(n_rows):
        row = list(confusion_matrix[r])
        correct.append(row.pop(r))
        wrong.append(sum(row))

    return correct, wrong


# correct = TRUE POSITIVES
# wrong = FALSE POSITIVES
def _get_correct_and_wrong_pixels_columns(confusion_matrix):
    correct = []
    wrong = []

    n_cols = confusion_matrix.shape[1]
    # for every row

    for c in range(n_cols):
        row = list(confusion_matrix[:, c])
        correct.append(row.pop(c))
        wrong.append(sum(row))

    return correct, wrong
