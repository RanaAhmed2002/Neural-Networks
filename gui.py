import tkinter as tk
import Algorithms

window = tk.Tk()
window.title('task 1')
neuron_entries = []
neuron_values = []
Network = Algorithms.initialize_network(5,[0,1], 3,0)
roundness_var = tk.StringVar()
perimeter_var = tk.StringVar()
MIN_var = tk.StringVar()
MXAL_var = tk.StringVar()
area_var = tk.StringVar()

###################################TESTING GUI ###############################
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
    perimeter_entry = tk.Entry(window2,textvariable=perimeter_var)
    perimeter_entry.pack()

    MXAL_label = tk.Label(window2, text='enter MajorAxisLength:')
    MXAL_label.pack()
    MXAL_entry = tk.Entry(window2,textvariable=MXAL_var)
    MXAL_entry.pack()


    MIN_label = tk.Label(window2, text='Enter MinorAxisLength:')
    MIN_label.pack()
    MIN_entry = tk.Entry(window2,textvariable=MIN_var)
    MIN_entry.pack()

    roundness_label = tk.Label(window2, text='Enter roundness:')
    roundness_label.pack()
    roundness_entry = tk.Entry(window2,textvariable=roundness_var)
    roundness_entry.pack()

    sample_button = tk.Button(window2, text='make predictions', command=main2)
    sample_button.pack()


def main2():
    global Network
    num_layers = int(No_Layers_entry.get())
    selected_activation = activation_var.get()
    bias_check = checkbox_var.get()
    inputs1 = [float(area_var.get()), float(perimeter_var.get()), float(MXAL_var.get()), float(MIN_var.get()), float(roundness_var.get())]
    prediction = Algorithms.make_pred(Network,inputs1,selected_activation,num_layers,bias_check)
    Algorithms.print_orginal(prediction)
    print("Prediction:", prediction)


###################################################################################################################


def create_neuron_inputs():
    global neuron_entries
    neuron_entries = []  # Store input fields for neurons

    num_layers = int(No_Layers_entry.get())  # Get the number of layers from the user input

    for i in range(num_layers):
        layer_label = tk.Label(window, text=f"Layer {i + 1} neurons:")
        layer_label.pack()

        neuron_entry = tk.Entry(window)
        neuron_entry.pack()

        neuron_entries.append(neuron_entry)  # Store the entry for later use

    submit_button = tk.Button(window, text="Submit , Initialize And Train ", command=Submit_and_train_func)
    submit_button.pack()
    #print(neuron_entries)
    return neuron_entries


def Submit_and_train_func():
    global neuron_entries
    global neuron_values
    global Network  # Store the network
    selected_activation = activation_var.get()
    bias_check = checkbox_var.get()
    epochs= int(epochs_var.get())
    learning= float(learning_rate_var.get())
    neuron_values = [int(entry.get()) for entry in neuron_entries]
    Network = Algorithms.initialize_network(5, neuron_values, 3,bias_check)
    Algorithms.main(Network,epochs,learning,neuron_values,selected_activation,bias_check)
    main()
    print("Neuron values entered:", neuron_entries)
    print("Neuron values:", neuron_values)



label = tk.Label(window, text='-----USER INPUT-----')
label.pack()

learning_label = tk.Label(window, text='Enter learning rate:')
learning_label.pack()
learning_rate_var = tk.StringVar()
learning_entry = tk.Entry(window, textvariable=learning_rate_var)
learning_entry.pack()

epochs_label = tk.Label(window, text='Enter epochs:')
epochs_label.pack()
epochs_var = tk.StringVar()
epochs_entry = tk.Entry(window, textvariable=epochs_var)
epochs_entry.pack()

checkbox_var = tk.IntVar()  # IntVar is used to store the checkbox value (0 for unchecked, 1 for checked)
biasedcheckbox = tk.Checkbutton(window, text="Biased", variable=checkbox_var)
biasedcheckbox.pack()

activation_var = tk.StringVar()
activation_label = tk.Label(window, text='-----CHOOSE ALGO-----')
activation_label.pack()

TanH = tk.Radiobutton(window, text='Hyperbolic Tangent sigmoid', variable=activation_var, value='TanH')
TanH.pack()

Sigmoid = tk.Radiobutton(window, text='Sigmoid', variable=activation_var, value='sigmoid')
Sigmoid.pack()

No_Layers_label = tk.Label(window, text="Enter the number of layers:")
No_Layers_label.pack()

No_Layers_entry = tk.Entry(window)
No_Layers_entry.pack()

submit_layers_button = tk.Button(window, text="Submit", command=create_neuron_inputs)
submit_layers_button.pack()
window.mainloop()
