import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import Evaluation
b = np.random.rand(1, 1) * 0.1  # bias


def fetch_data():
    dry_beans = pd.read_excel('C:/Users/Dragonoid/Desktop/Task 1 Final/Dry_Bean_Dataset.xlsx')
    dry_beans = dry_beans.ffill()
    return dry_beans

def split_data(data, classes, train_size=30, test_size=20):
    # Filter data for the selected classes
    selected_data = data[data['Class'].isin(classes)]
    class_1 = classes[0]
    class_2 = classes[1]
    # Get indices for the selected classes
    class_1_indices = selected_data[selected_data['Class'] == class_1].index
    class_2_indices = selected_data[selected_data['Class'] == class_2].index

    # Take the first 'train_size' rows for each class for training
    train_class_1 = selected_data.loc[class_1_indices[:train_size]]
    train_class_2 = selected_data.loc[class_2_indices[:train_size]]

    # Take the remaining rows after 'train_size' for each class for testing
    test_class_1 = selected_data.loc[class_1_indices[train_size:train_size + test_size]]
    test_class_2 = selected_data.loc[class_2_indices[train_size:train_size + test_size]]

    # Concatenate train and test data for the selected classes
    train_data = pd.concat([train_class_1, train_class_2])
    test_data = pd.concat([test_class_1, test_class_2])

    return train_data,test_data  # return the final data


def update_labels(df, classes):
    if classes==['BOMBAY','CALI']:
        label_mapping = {'BOMBAY': -1, 'CALI': 1, 'SIRA': 'SIRA'}  # Customize this mapping as needed
    elif classes==['BOMBAY', 'SIRA']:
        label_mapping = {'BOMBAY': -1, 'CALI': 'CALI', 'SIRA': 1}
    elif classes == ['CALI','SIRA']:
        label_mapping = {'BOMBAY': 'BOMBAY', 'CALI': -1, 'SIRA': 1}

    df['Class'] = df['Class'].map(label_mapping)
    return df

def visualize(X, y):
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.Paired, marker='o')
    plt.xlabel('MinorAxisLength')
    plt.ylabel('Roundnes')
    plt.show()


def plot_decision_boundary(features, classes, weight, bias, X_train_scaled, y_train):
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=plt.cm.Spectral)

    # Adjust axis limits
    plt.xlim(X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1)
    plt.ylim(X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1)

    # Plot the decision boundary
    plot_x = np.array([np.min(X_train_scaled[:, 0]) - 1, np.max(X_train_scaled[:, 0]) + 1])
    plot_y = (-1 / weight[1]) * (weight[0] * plot_x + bias)

    plt.plot(plot_x, plot_y, color='k', linestyle='--', linewidth=2)

    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def SLP_train_n_draw_n_test(features, classes, epochs, learning_rate, biased_checker=0):
    df = fetch_data()
    train_data, test_data = split_data(fetch_data(), classes)
    
    train_data = update_labels(train_data, classes)
    test_data = update_labels(test_data, classes)

    X_train = train_data[features]
    y_train = train_data['Class']
    X_test = test_data[features]
    y_test = test_data['Class']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    W = np.random.rand(len(features)) * 0.1
    b = np.random.rand() * 0.1 if biased_checker == 1 else 0

    for epoch in range(epochs):
        for i in range(len(X_train_scaled)):
            x = X_train_scaled[i]
            d = y_train.iloc[i]
            v = np.dot(W, x) + b
            y = np.sign(v)
            error = d - y
            e= int(error)
            if e != 0:
                 W += learning_rate * e * x
                 b += learning_rate * e if biased_checker==1 else 0

    print("Trained Weights:", W)
    print("Trained Bias:", b)
    Evaluation.test_confusion_matrix(W, y_test, X_test_scaled)
    plot_decision_boundary(features, classes, W, b, X_train_scaled, y_train)

    return W, b

# Example usage:
#SLP_train_n_draw_n_test(['Area', 'Perimeter'], ['CALI', 'SIRA'],10 , 0.01 , 0)

def adaline(features, classes, epochs, rate, mse_threshold, bias_checker=0):
    df = fetch_data()
    train_data, test_data = split_data(df, classes)
    
    train_data = update_labels(train_data, classes)
    test_data = update_labels(test_data, classes)

    X_train = train_data[features]
    y_train = train_data['Class']
    X_test = test_data[features]
    y_test = test_data['Class']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    w1 = np.random.rand()
    w2 = np.random.rand()
    weight = [w1, w2]

    b = np.random.rand() * 0.1 if bias_checker == 1 else 0

    for epoch in range(epochs):
        mean_squared_error = 0

        for i in range(len(X_train_scaled)):
            x = X_train_scaled[i]
            Target = y_train.iloc[i]

            NetValue = np.dot(weight, x) + b
            error = Target - NetValue

            weight += rate * error * x
            b += rate * error

            mean_squared_error += error ** 2

        mse = mean_squared_error / (2 * len(X_train_scaled))
        print(f"Epoch {epoch + 1}, MSE: {mse}")

        if mse < mse_threshold:
            print(f"MSE below threshold ({mse_threshold}). Stopping training.")
            break

    Evaluation.test_confusion_matrix(weight, y_train, X_train_scaled)
    Evaluation.test_confusion_matrix(weight, y_test, X_test_scaled)
    plot_decision_boundary(features, classes, weight, b, X_train_scaled, y_train)

    return weight, b

#adaline(['Area', 'roundnes'], ['CALI', 'SIRA'],100 , 0.01 ,0.3,0)
