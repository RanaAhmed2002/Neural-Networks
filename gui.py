import tkinter as tk
import numpy as np
import algorithms
import Evaluation
from sklearn.preprocessing import StandardScaler



# to allow for the choosing of two unique features
def update_second_feature_menu(*args):
    selected_feature_1 = feature_var_1.get()

    # Update the available features for the second option menu
    remaining_features = [f for f in features if f != selected_feature_1]
    menu = feature_menu_2["menu"]
    menu.delete(0, "end")
    for feature in remaining_features:
        menu.add_command(label=feature, command=lambda f=feature: feature_var_2.set(f))
    feature_var_2.set(remaining_features[0] if remaining_features else "")

def testonly():
    selected_features = [feature_var_1.get(),feature_var_2.get()]
    classes = option_var.get()
    non_formatted = classes.split(" & ")
    selected_classes = [name.strip() for name in non_formatted]
    selected_classes = [name.strip() for names in selected_classes for name in names.split(',')]
    learning_rate = float(learning_rate_var.get())
    epochs_selected = int(epochs_var.get())
    mse_thresholdd = float(mse_var.get())
    bias_var = checkbox_var.get()
    if algo_var.get() == "single":
        w,bias=algorithms.SLP_train_n_draw_n_test(selected_features, selected_classes, epochs_selected, learning_rate,bias_var)
    else:
        w,bias=algorithms.adaline(selected_features,selected_classes,epochs_selected,learning_rate,mse_thresholdd,bias_var)
    return w, bias

def algo_choice():
    selected_features = [feature_var_1.get(),feature_var_2.get()]
    classes = option_var.get()
    non_formatted = classes.split(" & ")
    selected_classes = [name.strip() for name in non_formatted]
    selected_classes = [name.strip() for names in selected_classes for name in names.split(',')]
    learning_rate = float(learning_rate_var.get())
    epochs_selected = int(epochs_var.get())
    mse_thresholdd = float(mse_var.get())
    bias_var = checkbox_var.get()
    if algo_var.get() == "single":
        w,bias=algorithms.SLP_train_n_draw_n_test(selected_features, selected_classes, epochs_selected, learning_rate,bias_var)
        inputs= get_inputs_from_window2()

        scaler = StandardScaler()
        inputs_scaled = scaler.fit_transform(inputs)
        zero_std_features = np.where(inputs_scaled.std(axis=0) == 0)[0]

        for feature_idx in zero_std_features:
            inputs_scaled[:, feature_idx] = np.random.normal(0, 1e-5, inputs_scaled.shape[0])

        print(inputs_scaled)
        weighted_sum=0
        weighted_sum = np.dot(inputs_scaled,w.T) + bias
        print(weighted_sum)
        res= np.sign(weighted_sum)
        print(float(res))
        if res >= 0:
            result= selected_classes[1]
        else:
            result= selected_classes[0]
        print(result)
        #return W,b
    else:
        w,bias=algorithms.adaline(selected_features,selected_classes,epochs_selected,learning_rate,mse_thresholdd,bias_var)
        inputs= get_inputs_from_window2()

        scaler = StandardScaler()
        inputs_scaled = scaler.fit_transform(inputs)
        zero_std_features = np.where(inputs_scaled.std(axis=0) == 0)[0]

        for feature_idx in zero_std_features:
            inputs_scaled[:, feature_idx] = np.random.normal(0, 1e-5, inputs_scaled.shape[0])

        print(inputs_scaled)
        weighted_sum=0
        weighted_sum = np.dot(inputs_scaled,w.T) + bias
        print(weighted_sum)
        res= np.sign(weighted_sum)
        print(float(res))
        if res >= 0:
            result= selected_classes[1]
        else:
            result= selected_classes[0]
        print(result)
    return w, bias
    

def main():
    # Main.visualize()
    window2 = tk.Toplevel()
    window2.title('enter sample')
    window2.geometry("300x300")

    area_label = tk.Label(window2, text='Enter area:')
    area_label.pack()
    area_entry = tk.Entry(window2,textvariable=area_var)
    area_entry.pack()

    perimeter_label = tk.Label(window2, text='enter perimeter:')
    perimeter_label.pack()
    perimeter_entry = tk.Entry(window2,textvariable=per_var)
    perimeter_entry.pack()

    MXAL_label = tk.Label(window2, text='enter MajorAxisLength:')
    MXAL_label.pack()
    MXAL_entry = tk.Entry(window2,textvariable=max_var)
    MXAL_entry.pack()

    MIN_label = tk.Label(window2, text='Enter MinorAxisLength:')
    MIN_label.pack()
    MIN_entry = tk.Entry(window2,textvariable=min_var)
    MIN_entry.pack()

    roundness_label = tk.Label(window2, text='Enter roundness:')
    roundness_label.pack()
    roundness_entry = tk.Entry(window2,textvariable=round_var)
    roundness_entry.pack()

    sample_button = tk.Button(window2, text='Test, Show Accuracy And Confusion Matrix', command=algo_choice)
    sample_button.pack()


# hena functions button 2
def get_inputs_from_window2():
   area= float(area_var.get())
   perimeter= float(per_var.get())
   maxl=float(max_var.get())
   minl=float(min_var.get())
   round=float(round_var.get())
   inputs=[]

   if feature_var_1.get()=='Area':
       x=area
       inputs.append(x)
   elif feature_var_1.get()=='Perimeter':
       x=perimeter
       inputs.append(x)
   elif feature_var_1.get()=='MajorAxisLength':
       x=maxl
       inputs.append(x)
   elif feature_var_1.get()=='MinorAxisLength':
       x=minl
       inputs.append(x)
   elif feature_var_1.get()=='roundnes':
       x=round
       inputs.append(x)

   if feature_var_2.get()=='Area':    
       x2=area
       inputs.append(x2)
   elif feature_var_2.get()=='Perimeter':   
       x2=perimeter
       inputs.append(x2)
   elif feature_var_2.get()=='MajorAxisLength':   
       x2=maxl
       inputs.append(x2)
   elif feature_var_2.get()=='MinorAxisLength':  
       x2=minl
       inputs.append(x2)
   elif feature_var_2.get()=='roundnes':   
       x2=round
       inputs.append(x2)

   inputs_array = np.array(inputs).reshape(1, -1)
 
   return inputs_array


window = tk.Tk()
window.title('task 1')
window.geometry("500x400")

area_var=tk.StringVar(window)
per_var=tk.StringVar(window)
max_var=tk.StringVar(window)
min_var=tk.StringVar(window)
round_var=tk.StringVar(window)

label = tk.Label(window, text='-----USER INPUT-----')
label.grid(row=0, column=1, columnspan=2, pady=10)

# Create a label for the first feature selection
feature1_label = tk.Label(window, text='First feature:')
feature1_label.grid(row=1, column=0, pady=5)

feature_var_1 = tk.StringVar()
feature_var_1.set("Area")  # Set the default feature

# Create the first option menu
features = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "roundnes"]
feature_menu_1 = tk.OptionMenu(window, feature_var_1, *features)
feature_menu_1.grid(row=1, column=1, pady=5)

feature_var_1.trace("w", update_second_feature_menu)

# Create a label for the second feature selection
label_2 = tk.Label(window, text="Select Feature 2:")
label_2.grid(row=2, column=0, pady=5)

# Create a variable to store the selected feature for the second option menu
feature_var_2 = tk.StringVar()
feature_var_2.set("Perimeter")  # Set the default feature

# Create the second option menu
feature_menu_2 = tk.OptionMenu(window, feature_var_2, *features)
feature_menu_2.grid(row=2, column=1, pady=5)

# class selection
class_selection_label = tk.Label(window, text='Choose Classes:')
class_selection_label.grid(row=3, column=0, pady=5)
option_var = tk.StringVar()
option_var.set("BOMBAY & CALI")  # Set the default option
options = ["BOMBAY & CALI", "BOMBAY & SIRA", "CALI & SIRA"]
option_menu = tk.OptionMenu(window, option_var, *options)
option_menu.grid(row=3, column=1, pady=5)

learning_label = tk.Label(window, text='Enter Learning Rate:')
learning_label.grid(row=1, column=2, pady=5)
learning_rate_var = tk.StringVar()
learning_entry = tk.Entry(window, textvariable=learning_rate_var)
learning_entry.grid(row=1, column=3, pady=5)

mse_label = tk.Label(window, text='Enter MSE Threshold:')
mse_label.grid(row=3, column=2, pady=5)
mse_var = tk.StringVar()
mse_entry = tk.Entry(window, textvariable=mse_var)
mse_entry.grid(row=3, column=3, pady=5)

checkbox_var = tk.IntVar()  # IntVar is used to store the checkbox value (0 for unchecked, 1 for checked)
biasedcheckbox = tk.Checkbutton(window, text="Biased", variable=checkbox_var)
biasedcheckbox.grid(row=4, column=2, pady=5)

algo_var = tk.StringVar()

algo_label = tk.Label(window, text='-----CHOOSE ALGO-----')
algo_label.grid(row=5, column=1, columnspan=2, pady=10)

epochs_label = tk.Label(window, text='Enter Number of Epochs:')
epochs_label.grid(row=2, column=2, pady=5)
epochs_var = tk.StringVar()
epochs_entry = tk.Entry(window, textvariable=epochs_var)
epochs_entry.grid(row=2, column=3, pady=5)

single = tk.Radiobutton(window, text='Single Layer Perceptron', variable=algo_var, value='single')
single.grid(row=6, column=0, columnspan=2, pady=5)

adaline = tk.Radiobutton(window, text='Adaline', variable=algo_var, value='adaline')
adaline.grid(row=6, column=2, columnspan=2, pady=5)

sample_button = tk.Button(window, text='Test Entries', command=testonly)
sample_button.grid(row=7, column=1, columnspan=2, pady=10)


button = tk.Button(window, text='Enter Sample & Train', command=main)
button.grid(row=8, column=1, columnspan=2, pady=10)

window.mainloop()
