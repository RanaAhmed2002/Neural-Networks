import pandas as pd
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
import Evaluation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def fetch_data():
    dry_beans = pd.read_excel('Dry_Bean_Dataset.xlsx')
    dry_beans = dry_beans.ffill()
    return dry_beans


def update_labels(df, classes):
    label_mapping = {classes[0]: -1, classes[1]: 2, classes[2]: 1}
    df['Class'] = df['Class'].map(label_mapping)
    return df


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs, bias_check):
    # n_inputs -> features(5), n_hidden -> array of num of neurons in each layer, n_outputs -> classes(3)
    network = list()
    for i in range(len(n_hidden)):
        # Use n_hidden[i] to get the number of neurons in the current layer
        if bias_check==1:
            hidden_layer = [{'weights_hidden_neurons': [random() for _ in range(n_inputs + 1)]} for _ in range(n_hidden[i])]
        else:
            hidden_layer = [{'weights_hidden_neurons': [random() for _ in range(n_inputs)]} for _ in range(n_hidden[i])]
        network.append(hidden_layer)
        n_inputs=n_hidden[i]
 
    # Use n_hidden[-1] to get the number of neurons in the last hidden layer
    last_layer = n_hidden[-1]
    if bias_check==1:
        output_layer = [{'weights_output_neurons': [random() for _ in range(last_layer + 1)]} for _ in range(n_outputs)]
    else:
        output_layer = [{'weights_output_neurons': [random() for _ in range(last_layer)]} for _ in range(n_outputs)]
    network.append(output_layer)
    for layer in network:
        print("Layers:")
        print(layer)
    return network


def calc_net(weights, inputs,bias_check=0):
    net = 0
    if bias_check ==0:
        for i in range(len(weights)):
            net += float(weights[i]) * float(inputs[i])
    else:
        for i in range(len(weights)-1):
            net += float(weights[i]) * float(inputs[i])
    return net

def calc_net2(weights, inputs, bias):
    return np.dot(inputs, weights) + bias

def forward_propagate(network, row, selected_activation, n_hidden, bias_check):
    activations = []
    # Forward propagate through each hidden layer ma3ada el output layer
    input_data = row
    for layer in network[:-1]:
        new_inputs = []
        for neuron in layer:
            weights = neuron['weights_hidden_neurons']
            net = calc_net(weights, input_data, bias_check)
            if selected_activation == 'sigmoid':
                neuron['output'] = a = sigmoid(net)
                activations.append(a)
            else:
                neuron['output'] = a = hypertan(net)
                activations.append(a)
            new_inputs.append(neuron['output'])
        input_data = new_inputs

    # to get the weights of each neuron in the last layer 3ashan da hyb2a el input beta3 el output layer
    last_layer_act = []
    reversed_act = activations
    reversed_act.reverse()

    if isinstance(n_hidden, int):
        for i in range(n_hidden):
            input_val = reversed_act[i]
            last_layer_act.append(input_val)
    elif isinstance(n_hidden, list):
        for i in range(n_hidden[-1]):
            input_val = reversed_act[i]
            last_layer_act.append(input_val)

    # Forward propagate through the output layer
    output_layer = network[-1]
    output_activations = []
    for neuron in output_layer:
        weights = neuron['weights_output_neurons'][:-1]
        net = calc_net(weights, last_layer_act, bias_check)
        if selected_activation == 'sigmoid':
            neuron['output'] = a = sigmoid(net)
            output_activations.append(a)
        else:
            neuron['output'] = a = hypertan(net)
            output_activations.append(a)
    print("output_activations",output_activations)
    return output_activations, activations


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def hypertan(z):
    return np.tanh(z)


# def transfer_derivative(output):
#     return output * (1.0 - output)
def transfer_derivative(output):
    if isinstance(output, list):
        output = output[0]  # Assuming the list contains a single value
    return output * (1.0 - output)




def backward_propagation(network, expected_output, activations, learning_rate, bias_check, output_activations):

    errors=[ ]
    for i in reversed(range(len(network))):
        print("i",i)
        layer = network[i]
        print("layer:",layer)
        # Compute errors for each neuron in the layer
        if i != len(network)-1:
            print("in hidden layer")
            #print("error before", errors )
            for j in range(len(layer)):
                print("======================NEW LAYER==================")
                for neuron in network[i]:
                    error = 0.0
                    neuron['error'] = errors[-1]
                    print("neuron:", neuron)
                    print("length network[i+1]",len(network[i+1]))
                    weights=neuron['weights_hidden_neurons']
                    output = neuron['output']
                    e = neuron['error']
                    for j in range(len(weights)):
                        print(j)
                        print(weights[j])
                        print("current error:", e)
                        error += (weights[j]* e)
                    final_error= error*transfer_derivative(output)

                #errors = []   
                errors.append(final_error)
            
        else:
            # Output layer error signal
            print("in output layer")
            print(len(layer))
            for j in range(len(layer)):
                print(j)
                node = layer[j]
                error = (output_activations[j] - expected_output[j])*transfer_derivative(output_activations[j])
                node['error']=error
                print(error)
                errors.append(error)
                print(node)
        print("errors after:",errors)

        # Update error signals and weights
    for j in range(len(layer)):
            neuron = layer[j]
            errors_sum = sum(errors)
            neuron['error_signal'] = errors_sum * transfer_derivative(neuron['output'])

            # Update weights for hidden and output layers
                    # Update weights for hidden and output layers
            if 'weights_hidden_neurons' in neuron:
                weights_key = 'weights_hidden_neurons'
                activations_for_update = activations[i - 1]
            elif 'weights_output_neurons' in neuron:
                weights_key = 'weights_output_neurons'
                activations_for_update = activations[i]

            if weights_key in neuron and isinstance(neuron[weights_key], list):
                for k in range(len(neuron[weights_key]) - 1):
                    if k < activations_for_update:
                        neuron[weights_key][k] -= learning_rate * neuron['error_signal'] * activations_for_update
                    # else:
                    #     print(f"Index {k} is out of range for activations_for_update.")
            else:
                print(f"Neuron does not have {weights_key} or it is not a list.")

    return network

def der_sigmoid(self, sigmoid):
    return sigmoid * (1 - sigmoid)

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def der_tanh(tanh):
    return 1 - (tanh * tanh)


def split_data(data, train_size=30, test_size=20):
    Classes = data['Class'].unique()
    # Get indices for the classes
    # print(Classes)
    class_1_indices = data[data['Class'] == Classes[0]].index
    class_2_indices = data[data['Class'] == Classes[1]].index
    class_3_indices = data[data['Class'] == Classes[2]].index

    # Take the first 'train_size' rows for each class for training
    train_class_1 = data.loc[class_1_indices[:train_size]]
    train_class_2 = data.loc[class_2_indices[:train_size]]
    train_class_3 = data.loc[class_3_indices[:train_size]]
    # Take the remaining rows after 'train_size' for each class for testing
    test_class_1 = data.loc[class_1_indices[train_size:train_size + test_size]]
    test_class_2 = data.loc[class_2_indices[train_size:train_size + test_size]]
    test_class_3 = data.loc[class_3_indices[train_size:train_size + test_size]]

    # Concatenate train and test data for the selected classes
    train_data = pd.concat([train_class_1, train_class_2, train_class_3])
    test_data = pd.concat([test_class_1, test_class_2, test_class_3])

    return train_data, test_data  # return the final data

def main(network, epochs, learning_rate, n_hidden, selected_activation, bias_check):
    df = fetch_data()
    df = update_labels(df, ['BOMBAY', 'CALI', 'SIRA'])
    traindf, testdf = split_data(df)

    X_test = testdf[['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']]
    X_train = traindf[['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']]  # Replace with your feature columns
    y_train = traindf['Class']
    y_test = testdf['Class']

    # print(X_train)
    # print(y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled)
    print("Initialize network:")
   
    X_train_scaled = pd.DataFrame(X_train_scaled)
    # bombay -> first -1
    # cali -> second 2
    # sira -> third 1 ... expected output for bombay example : [-1,0,0]... for cali : [0,2,0]...for sira [0,0,1]

    for epoch in range(epochs):
        for index in range(len(X_train_scaled)):
            row = X_train_scaled.iloc[index]
            expected = y_train.iloc[index]
            if expected == -1:
                expected=[-1,0,0]
            elif expected == 2:
                expected=[0,2,0]
            elif expected == 1:
                expected=[0,0,1]  
            print("expected value:", expected)
            output_activations,activation = forward_propagate(network, row, selected_activation ,n_hidden,bias_check)
            backward_propagation(network, expected, activation, learning_rate, bias_check, output_activations)
    y_pred = []
    for index in range(len(X_test_scaled)):
        row = X_test_scaled.iloc[index]
        predicted_class = make_pred(network, row, selected_activation, n_hidden, bias_check)
        y_pred.append(predicted_class)
    Evaluation.test_confusion_matrix(y_test, y_pred)



def print_orginal(predicted_class):
    if predicted_class == -1:
        predicted_class = 'bombay'
        print("bombay")
    elif predicted_class == 2:
        predicted_class = 'cali'
        print("cali")
    elif predicted_class == 1:
        predicted_class = 'sira'
        print("sira")
    else:
        predicted_class = 'unknown'
        print("unknown class")
def make_pred(network, input_data,selected_activation,n_hidden,bias_check):
        # Forward propagate through the pre-trained network
        output_activations,_ = forward_propagate(network, input_data,selected_activation,n_hidden,bias_check)

        # Assuming the output layer neurons represent class probabilities
        # Choose the class with the highest probability as the predicted class
        predicted_class = np.argmax(output_activations)
        print(predicted_class)
        if predicted_class == 0:
            predicted_class = -1
        elif predicted_class == 1:
            predicted_class = 2
        elif predicted_class == 2:
            predicted_class = 1
        return predicted_class

#network = initialize_network(5, [1,2], 3,0)
#main(network,10,0.1,[1,2], 'sigmoid',0)