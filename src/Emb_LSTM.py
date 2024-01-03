#!/usr/bin/env python


''' ---------------- About the script ----------------

Many-to-many LSTM neural network to predict the subsequent dance move in Lindy Hop. This product was build as part of the Natural language processing exam at Aarhus University.

Author: Ruta Slivkaite
Course instructor: Roberta Rocca


This script builds and trains a Many-to-many LSTM neural network for the Lindy Hop dance sequence prediction task. It aims to contribute to a better and more evidence-based understanding of the dance language.
The code could be re-used with any other dance dataset with the same structure. The dance move sequences should be saved as .csv file with a 'Behavior'column where each row is a separate dance sequence of labeled moves separated by a comma.

The script prints the evaluation metrics to the terminal, and saves classification report and training/validation accuracy/loss plots as well as the architecture of the model as .png files in the directory. 

Preprocessing of the data involves:
- encoding the dance moves to integers
- no padding of sequences were performed since the code takes into account the different length of each sequence. 
- preparing input-output pairs for LSTM
- spliting the data into training and testing sets (80-20 split)


Arguments:
    
    -dir,    --data_dir:         Directory of the CSV file with dance move sequences
    -batch,  --batch_size:       A subset of the training data used in one iteration of model training or inference. Default = 16 
    -epochs, --number_epochs:    Defines how many times the learning algorithm will work through the entire training dataset. Default = 1000
    -emb_m,  --embedding_matrix_path:  Directory of the pre-trained .npy embedding matrix       


Run the code example:    
    
    with default arguments:
        $ python Emb_LSTM.py -dir data/LindyHop_moves_sequences.csv -emb_m data/embedding_matrix_word2vec_100.npy
        
    with optional arguments:
        $ python Emb_LSTM.py -dir data/LindyHop_moves_sequences.csv -batch 32 -epochs 500 -emb_m data/embedding_matrix_word2vec_100.npy
        
       

'''




"""---------------- Importing libraries ----------------
"""
# system tools
import os
import sys
sys.path.append(os.path.join(".."))

# pandas, numpy, argparse, csv
import numpy as np
import pandas as pd
import argparse
import csv

# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, TimeDistributed, Dense
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# matplotlib
import matplotlib.pyplot as plt

# random, pickle, ast
import random
import pickle
import ast








"""---------------- Main script ----------------
"""

    
    
    
"""---------------- Functions ----------------
"""

def main():

    
    # Function to convert dance moves to integers
    def encode_sequence(sequence):
        # Encode the sequence
        return label_encoder.transform(sequence)    


    # Function for creating input-output pairs
    def create_input_output_pairs_mtm(data, time_steps):
        """
        Function to split the sequence data into input and target arrays.
        It maintains the sequence structure by looping through each sequence 
        (equivalent to one dance) and treating them separately.


        data: encoded dance sequences
        time_steps: number of dance moves to look back at to predict the subsequent moves
        """
        x, y = [], []
        for sequence in data:
            for i in range(len(sequence) - time_steps):
                x.append(sequence[i:(i + time_steps)])  # Input sequence of x moves
                y.append(sequence[(i + 1):(i + 1 + time_steps)])  # Output sequence of x moves

        return x, y
    

    def plot_history(H, epochs):
        """
        Function for plotting model history using matplotlib

        H: model history 
        epochs: number of epochs for which the model was trained
        """
        plt.style.use("fivethirtyeight")
        plt.figure()
        plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig('../out/LSTM_performance_graph.png')    

    
    
    """------ Argparse parameters ------
    """
    # Instantiating the ArgumentParser  object as parser 
    parser = argparse.ArgumentParser(description="[INFO] Train Many-To-Many LSTM model with an Embedding layer to predict subsequent dance move")
    
    # Adding arguments
    parser.add_argument("-dir", "--data_dir", required = True, help = "Directory of the CSV file")
    parser.add_argument("-batch", "--batch_size", required=False, type=int, default=16, help='Batch size for training (e.g., 16, 32, 64)')
    parser.add_argument("-epochs","--number_epochs", required=False, type=int, default=1000, help='Number of epochs for training')
    parser.add_argument("-emb_m", "--embedding_matrix_path", type=str, help='Path to the embedding matrix file') # .npy file
    
    
    # Parsing the arguments
    args = vars(parser.parse_args())
    ##args = parser.parse_args()
    
    
    # Saving parameters as variables
    data = args["data_dir"] # Directory of the CSV file
    batch_size = args["batch_size"] # Batch size
    epochs = args["number_epochs"] # Epochs
    embedding_matrix = args["embedding_matrix_path"] # Directory of the embedding matrix

    

   
    
    """------ Loading data ------
    """

    # Message to a user
    print("\n[INFO] Loading data and preparing for training a Many-to-Many LSTM model...")
    
    # Create ouput folder, if it doesn´t exist already, for saving the classification report, performance graph and model´s architecture 
    if not os.path.exists("../out"):
        os.makedirs("../out")
    
    # Loading and reading data
    filename = os.path.join(data)
    data = pd.read_csv(filename)
    
    # Convert strings back to lists
    data['Behavior'] = data['Behavior'].apply(ast.literal_eval)
    behavior_list_of_lists = data['Behavior'].tolist()
       
    
    
    """------ Preprocessing: encode the dance moves to integers ------
    """
    
    # Flatten all sequences into a single list to find unique moves
    all_moves = [move for sequence in behavior_list_of_lists for move in sequence]

    # Create and fit the LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(all_moves)

    # Number of unique classes (moves)
    num_classes = len(label_encoder.classes_)
    
    # Apply encoding to each sequence
    encoded_sequences = [encode_sequence(sequence) for sequence in behavior_list_of_lists]
    

    # Extract class labels
    class_labels = label_encoder.classes_
    

    
        
    """------ Prepare input-output pairs for LSTM ------
    """
 
    # create input-output pairs
    x_emb, y_emb = create_input_output_pairs_mtm(encoded_sequences, 4)
    
    # Convert an input to Numpy Array
    x_emb = np.array(x_emb)
    
    # Convert an output to a one-hot-encoded vector
    y_emb_one_hot = to_categorical(y_emb, num_classes=num_classes)
    
    
    
    # Split the data into training and testing sets (80-20 split)
    split_index_2 = int(0.8 * len(x_emb))
    x_emb_train, x_emb_val = x_emb[:split_index_2], x_emb[split_index_2:]
    y_emb_train, y_emb_val = y_emb_one_hot[:split_index_2], y_emb_one_hot[split_index_2:]

    # Check shapes
    print("Shape of x_emb_train:", x_emb_train.shape)
    print("Shape of y_emb_train:", y_emb_train.shape)
    

            
    """------ Define the LSTM model with the Embedding layer ------
    """
    
# Build the model
def build_model(num_moves, embedding_dim, input_length, embedding_matrix):
    model = Sequential([
        Embedding(input_dim=num_moves, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False, 
                  input_length=input_length),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        TimeDistributed(Dense(num_moves, activation='softmax'))
    ])
    return model
    
    
    # load embedding matrix
    embedding_matrix = np.load(embedding_matrix)
    
    
    # Model parameters
    input_length = 4  # Length of input sequences
    num_moves = len(embedding_matrix)   # Number of unique moves or classes
    embedding_dim = len(embedding_matrix[0])  # Dimension of Word2Vec embeddings

    
    """------ Compile the model ------
    """    
    
    Emb_model = build_model(num_moves, embedding_dim, input_length, embedding_matrix)
    Emb_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    # Print summary
    Emb_model.summary()
    
    # Ploting and saving model´s architecture
    plot_model(Emb_model, to_file='../out/LSTM´s_Model´s_architecture.png',
               show_shapes=True,
               show_dtype=True,
               show_layer_names=True)
    
    # Printing that model´s architecture graph has been saved
    print(f"\n[INFO] LSTM´s model´s architecture graph has been saved")
    
    
    """------ Train the model ------
    """    
    
    print("[INFO] training and evaluating LSTM model ...")         
    history_emb = Emb_model.fit(x_emb_train, y_emb_train, epochs=epochs, batch_size=batch_size, validation_data=(x_emb_val, y_emb_val))    
    
    # Save the training history
    history_file = "history_emb_baseline.pkl"
    with open(history_file, 'wb') as file:
        pickle.dump(history_emb.history, file)
    

    """------ Evaluate the model ------
    """    
            
# Functionto xtract and print metrics
def print_metrics(history):
    val_accuracy_per_epoch = history.history['val_accuracy']
    train_accuracy_per_epoch = history.history['accuracy']
    train_loss_per_epoch = history.history['loss']
    val_loss_per_epoch = history.history['val_loss']

    best_epoch_val_accuracy = val_accuracy_per_epoch.index(max(val_accuracy_per_epoch)) + 1
    best_epoch_train_accuracy = train_accuracy_per_epoch.index(max(train_accuracy_per_epoch)) + 1
    best_epoch_train_loss = train_loss_per_epoch.index(min(train_loss_per_epoch)) + 1
    best_epoch_val_loss = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1

    print(f"Best Epoch for Validation Accuracy: {best_epoch_val_accuracy} (Val Accuracy: {max(val_accuracy_per_epoch)})")
    print(f"Best Epoch for Training Accuracy: {best_epoch_train_accuracy} (Train Accuracy: {max(train_accuracy_per_epoch)})")
    print(f"Best Epoch for Training Loss: {best_epoch_train_loss} (Train Loss: {min(train_loss_per_epoch)})")
    print(f"Best Epoch for Validation Loss: {best_epoch_val_loss} (Val Loss: {min(val_loss_per_epoch)})")
    print("\nOverall Best Performance Metrics:")
    print(f"Maximum Validation Accuracy: {max(val_accuracy_per_epoch)}")
    print(f"Maximum Training Accuracy: {max(train_accuracy_per_epoch)}")
    print(f"Minimum Training Loss: {min(train_loss_per_epoch)}")
    print(f"Minimum Validation Loss: {min(val_loss_per_epoch)}")

    
    print("LSTM evaluation:") 
    print_metrics(history_emb)
    
    # Save the performance graph
    plot_history(history, epochs = epochs) 

    # Printing that performance graph has been saved
    print(f"\n[INFO] LSTM´s performance graph has been saved")


    
    """------ Classification report ------
    """    
          
    # Predict classes on the test set
    y_pred = Emb_model.predict(x_emb_val)

    # Convert predictions from one-hot encoded back to label indices
    y_pred_classes = np.argmax(y_pred, axis=-1)
    y_true_classes = np.argmax(y_emb_val, axis=-1)

    # Convert numeric classes to actual labels
    y_pred_labels = [class_labels[i] for i in y_pred_classes.flatten()]
    y_true_labels = [class_labels[i] for i in y_true_classes.flatten()]


    # Classification report
    class_report = classification_report(y_true_labels, y_pred_labels, zero_division=0)
    print(class_report)


    # Defining full filepath to save .csv file 
    outfile = os.path.join("../", "out", "LSTM_classifier_report.csv")
    
    # Save the report
    with open(outfile, 'w') as file:
        file.write(class_report)
    
 

    print(f"\n[INFO] Classification report has been saved")

    
    print("\nScript was executed successfully! Have a nice day")
        

        

"""---------------- End of the main script ----------------
"""




        
# Define behaviour when called from command line
if __name__=="__main__":
    main()

