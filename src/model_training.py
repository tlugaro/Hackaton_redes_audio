import pandas as pd
import argparse
import numpy as np
import joblib

from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import tensorflow as tf

def load_data(file_path):
    df = pd.read_pickle(file_path)
    return df

def split_data(df):
    print(df)
    X_train, X_val, y_train, y_val = train_test_split(np.array(df["x"].tolist()), np.array(df["y"].tolist()), test_size=0.1, random_state=24)

    return X_train, X_val, y_train, y_val

def train_model(X_train,X_val,y_train, y_val):
    np.random.seed(21)
    tf.random.set_seed(21)
    num_clases=len(set(y_train))
    model = models.Sequential([
        layers.Input(shape=(128, 216)),
        layers.Conv1D(32, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.15),
        layers.LSTM(64),
        layers.Dropout(0.15),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_clases, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val))

    return model

def save_model(model, model_path,model_file2):
    model.save(model_path)
    joblib.dump(model,model_file2)
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model training script for Automated Instrument Sound Recognition Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/processed_data/data.pkl',
        help='Path to the processed data file to train the model'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.h5',
        help='Path to save the trained model'
    )
    parser.add_argument(
        '--model_file2',
        type=str,
        default='models/model.pkl',
        help='Path to save the trained model'
    )
    return parser.parse_args()

def main(input_file, model_file,model_file2):
    df = load_data(input_file)
    X_train, X_val, y_train, y_val = split_data(df)
    model = train_model(X_train, X_val, y_train, y_val)
    save_model(model, model_file,model_file2)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.model_file2)