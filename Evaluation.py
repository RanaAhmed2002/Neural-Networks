import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


def calculate_accuracy(true_pos, true_neg, false_pos, false_neg): #calculates the accuracy
    total = true_pos + true_neg + false_pos + false_neg
    accuracy = (true_pos + true_neg) / total
    return accuracy

def plot_confusion_matrix(y_true, y_pred ,true_pos , true_neg , false_pos , false_neg):

    unique_classes = np.unique(np.concatenate((y_true.ravel(), y_pred.ravel())))
    num_classes = len(unique_classes)

    confusion_matrix = np.zeros((num_classes, num_classes))

    for i in range(num_classes):
        for j in range(num_classes):
            confusion_matrix[i, j] = np.sum((y_true.ravel() == unique_classes[i]) & (y_pred.ravel() == unique_classes[j]))

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, unique_classes, rotation=45)
    plt.yticks(tick_marks, unique_classes)

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(confusion_matrix[i, j]), ha="center", va="center")
    accuracy = calculate_accuracy(true_pos, true_neg, false_pos, false_neg)
    # Display accuracy
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2%})')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def test_confusion_matrix(W, y_test, X_test):
    y_pred = np.sign(np.dot(X_test, W.T))  # Transpose W before dot product
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    predicted = np.where(y_pred >= 0, 1, -1)  # Apply threshold (0 in this case)

    for actual, pred in zip(y_test, predicted):
        if actual == pred:
            if actual == 1:
                true_pos += 1
            else:
                true_neg += 1
        else:
            if actual == 1:
                false_neg += 1
            else:
                false_pos += 1
    total = true_pos + true_neg + false_pos + false_neg
    #print(true_pos, true_neg, false_pos, false_neg)
    plot_confusion_matrix(y_test, predicted,true_pos , true_neg , false_pos , false_neg)
    return true_pos, true_neg, false_pos, false_neg












