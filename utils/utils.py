import csv
import pickle
import pandas as pd
from sentence_transformers.readers import InputExample

def open_csv(data_path):
    data_file = pd.read_csv(data_path)
    return data_file

def save_csv(save_data_path, data):
    data.to_csv(save_data_path, index=False)


def open_pkl(data_path):
    with open(data_path, 'rb') as pickle_file:
        data_file = pickle.load(pickle_file)
    return data_file


def save_pkl(save_data_path, data):
    print("saved ", save_data_path)
    with open(save_data_path, 'wb') as f:
        pickle.dump(data, f)


def get_input_samples(sub_sec_pairs):
    sample_list = [InputExample(texts=[sec_a_text, sec_b_text], label=label) 
    for sec_a_text, sec_b_text, label in zip(sub_sec_pairs['sec_a_text'], sub_sec_pairs['sec_b_text'], sub_sec_pairs['label'])]
    return sample_list

def save_test_val_result_csv(model_save_path, save_csv_result):
    val_df = open_csv(model_save_path+'/'+save_csv_result+'_val_results.csv')
    test_df = open_csv(model_save_path+'/'+save_csv_result+'_test_results.csv')
    final_df = pd.concat([val_df, test_df.iloc[:, 7:]], axis=1)
    final_df.to_csv(model_save_path+'/'+save_csv_result+'_final_results.csv', index=False)