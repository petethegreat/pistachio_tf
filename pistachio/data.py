'''
data.py
functions for working with data/datasets
'''

import pandas as pd 
from scipy.io import arff
import os 
from typing import List, Dict

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

def load_arff_file(input_arff: str, label_mapping: Dict[str:int]) -> pd.DataFrame:
    """convert arff file to parquet"""
    if not os.path.exists(input_arff):
        raise ValueError(f"input file '{input_arff}' does not exist")
    print(f'loading arff file {input_arff}')
    data, meta = arff.loadarff(input_arff)
    print(f"arff metadata: {meta}")
    df = pd.DataFrame(data)
    df['Class'] = df['Class'].astype(str).map(label_mapping)
    
    return df
##################

def split_csv_data(infilename: str, train_filename: str, test_filename:str, test_fraction: float):
    df = pd.read_csv(infilename, header=0)
    columns = df.columns
    df['split_var'] = np.random.uniform(size=len(df))
    train_df = df.loc[df.split_var <= test_fraction][columns]
    test_df = df.loc[df.split_var > test_fraction][columns]
    train_df.to_csv(train_filename, index=False, header=True)
    test_df.to_csv(test_filename, index=False, header=True)
    print(f'wrote {len(train_df)} records to {train_filename}')
    print(f'wrote {len(test_df)} records to {test_filename}')
##########################################################

def df_to_dataset(
        df: pd.DataFrame, 
        target_column: str,
        batch_size: int=32,
        prefetch: int=32,
        shuffle:bool=True, 
        drop:bool=True):
    feature_df = df.copy()
    target = feature_df.pop(target_column)
    dataset = tf.data.Dataset.from_tensor_slices((dict(feature_df), target))
    if shuffle:
        dataset=dataset.shuffle(buffer_size=len(feature_df))
    dataset = dataset\
        .batch(BATCH_SIZE, drop_remainder=drop)\
        .prefetch(PREFETCH)
    return dataset
##########################################################

def split_data_to_frames(infilename: str, test_fraction: float=0.2, val_fraction: float=0.25, seed:int=43, stratify=None):
    """ load csv as dataframe, split"""

    df = pd.read_csv(infilename)
    train_val_df, test_df = train_test_split(df, test_size=test_fraction, random_state=seed, stratify=df[stratify])
    train_df, val_df = train_test_split(train_val_df, test_size=val_fraction, random_state=seed+1, stratify=train_val_df[stratify])
    return train_df, val_df, test_df
##########################################################

def read_or_generate_splits(split_data_path: str, csv_filename: str, seed: int):
    ''' if we've previously generated split files, read them, else generate using current seed'''
    train_file = os.path.join(split_data_path, 'pistachio_train.csv')
    valid_file = os.path.join(split_data_path, 'pistachio_valid.csv')
    test_file = os.path.join(split_data_path, 'pistachio_test.csv')

    # read these files if they exist, else create and save splits
    if (
        os.path.exists(train_file) &
        os.path.exists(valid_file) &
        os.path.exists(test_file)):
            train_df = pd.read_csv(train_file, header=0)
            valid_df = pd.read_csv(valid_file, header=0)
            test_df = pd.read_csv(test_file, header=0)
    else:
        train_df, valid_df, test_df = split_data_to_frames(csv_filename, stratify='Class', seed=seed)
        train_df.to_csv(train_file, index=False, header=True)
        valid_df.to_csv(valid_file, index=False, header=True)
        test_df.to_csv(test_file, index=False, header=True)
        
    return train_df, valid_df, test_df
##########################################################   

def get_dataset_class_proportions(the_dataset):
    def count_class(counts, batch, num_classes=2):
        labels = batch[1] # class is second element of tuple
        for i in range(num_classes):
            cc = tf.cast(labels == i, tf.int32)
            counts[str(i)] += tf.reduce_sum(cc)
        return counts
    initial_state = {'0':0, '1':0}
    proportions = {k: v.numpy() for k,v in the_dataset.reduce(reduce_func=count_class, initial_state=initial_state).items()}
    total = sum(proportions.values())
    proportions.update({f'proportion_{k}': v/total for k,v in proportions.items()})
    
    return proportions