import argparse
import librosa
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import csv
def load_data(file_path):
    df = pd.read_csv(file_path)

    return df

def clean_data(df):
    df_clean=df.dropna()

    return df_clean
def preprocess_test_data(df):
    X_paths = df['Path'].tolist()

    idx = df['Idx'].tolist()
    num_classes = 6

    def preprocess_audio(file_path):
        y, sr = librosa.load(file_path)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec

    X_processed = [preprocess_audio("../"+file_path) for file_path in X_paths]
    X_processed = np.array(X_processed)
    X_processed_expanded = np.expand_dims(X_processed, axis=-1)
    scaler = StandardScaler()
    X_reshaped = X_processed_expanded.reshape((X_processed_expanded.shape[0], -1))
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape((X_processed_expanded.shape))
    df_processed = pd.DataFrame()
    df_processed["x"] = [X_scaled[i] for i in range(len(X_scaled))] 
    df_processed["idx"] = idx
    return df_processed
def preprocess_data(df):
    
    X_paths = df['Path'].tolist()

    y_labels = df['Label'].tolist()
    num_classes = 6

    def preprocess_audio(file_path):
        y, sr = librosa.load(file_path)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec

    X_processed = [preprocess_audio("../"+file_path) for file_path in X_paths]
    X_processed = np.array(X_processed)
    y = np.array(y_labels)

    X_processed_expanded = np.expand_dims(X_processed, axis=-1)
    scaler = StandardScaler()

    X_reshaped = X_processed_expanded.reshape((700, -1))
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape((X_processed_expanded.shape))

    df_processed = pd.DataFrame()
    df_processed["x"] = [X_scaled[i] for i in range(len(X_scaled))] 
    df_processed["y"] = y

    return df_processed

def save_data(df, output_file):
    df.to_pickle(output_file)
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Automated Instrument Sound Recognition Hackathon')
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/labels_paths_train.csv',
        help='Path to the raw data file to process'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='data/processed_data/data.pkl',
        help='Folder path to save the processed data'
    )
    parser.add_argument(
        '--test_input_file',
        type=str,
        default='data/paths_test.csv',
        help='Path to the raw data file to process'
    )
    parser.add_argument(
        '--test_output_file', 
        type=str, 
        default='data/processed_data/test_data.pkl',
        help='Folder path to save the processed data'
    )
    return parser.parse_args()

def main(input_file, output_file,test_input_file,test_output_file):
    df = load_data(input_file)
    df_clean = clean_data(df)
    df_processed = preprocess_data(df_clean)
    save_data(df_processed, output_file)
    test_df = load_data(test_input_file)
    test_df_clean = clean_data(test_df)
    test_df_processed = preprocess_test_data(test_df_clean)
    save_data(test_df_processed, test_output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.output_file, args.test_input_file,args.test_output_file)