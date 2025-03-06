from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def convert_labels_list(labels):
    return [1 if label == 'Yes' else 0 for label in labels]

def calculate_classification_metrics(pred, ground_truth, print_show = False, yesno = True):
    #label_format: 'yesno' or '1-0'
    # Calculate True Positives, True Negatives, False Positives, False Negatives
    # Yes: 1, No: 0
    if yesno:
        pred = convert_labels_list(pred)
        ground_truth = convert_labels_list(ground_truth)

    TP = sum([1 for p, gt in zip(pred, ground_truth) if p == 1 and gt == 1])
    TN = sum([1 for p, gt in zip(pred, ground_truth) if p == 0 and gt == 0])
    FP = sum([1 for p, gt in zip(pred, ground_truth) if p == 1 and gt == 0])
    FN = sum([1 for p, gt in zip(pred, ground_truth) if p == 0 and gt == 1])

    # Calculate TPR, TNR, FPR, FNR, Prediction, Accuracy, Recall, F1 Score
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) != 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) != 0 else 0
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    F1 = f1_score(ground_truth, pred)
    
    #Accuracy = accuracy_score(ground_truth, pred)
    #Recall = recall_score(ground_truth, pred) #TPR
    
    if print_show:
        print("True Positive Rate (TPR):", TPR)
        print("True Negative Rate (TNR):", TNR)
        print("False Positive Rate (FPR):", FPR)
        print("False Negative Rate (FNR):", FNR)
        print("Accuracy:", Accuracy)
        print("F1 Score:", F1)
    
    return TPR, TNR, FPR, FNR, Accuracy, F1

# Example usage
#pred = [1, 0, 1, 0, 1]
#ground_truth = [1, 1, 0, 0, 1]

#TPR, TNR, FPR, FNR, Prediction, Accuracy, Recall, F1 = calculate_classification_metrics(pred, ground_truth, print_show=True, yesno = False)