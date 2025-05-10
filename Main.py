from sklearn.model_selection import train_test_split
import algorithms
import gui

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#function for visualization
#def visualize(model,features,bias_var):
#GUI calling and Data Extraction

gui.main()
#splitting classes names
classes = gui.option_var.get()
non_formatted = classes.split(" & ")
selected_classes = [name.strip() for name in non_formatted ]
selected_classes = [name.strip() for names in selected_classes for name in names.split(',')]
#getting rest of the data
algo_option = gui.algo_var.get()
bias_var = gui.checkbox_var.get()
selected_features = [gui.feature_var_1.get(),gui.feature_var_2.get()]
learning_rate = float(gui.learning_rate_var.get())
epochs_selected = int(gui.epochs_var.get())

#SLP_train_n_draw(features, classes, epochs, learning_rate, biasedchecker=0):
def algo_choice():
    if algo_option == "single":
        w = algorithms.SLP_train_n_draw_n_test(selected_features, selected_classes, epochs_selected, learning_rate,bias_var)
    else:
        print("algorithm not implemented")


