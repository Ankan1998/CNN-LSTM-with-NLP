import pandas as pd
import random
from torchtext.legacy.data import TabularDataset
from dataset_processing.custom_field import field_maker

def train_test_maker(train_file_name,test_file_name):
    field_train = field_maker()
    train_data = TabularDataset(
        path=train_file_name,
        format='csv',
        fields=field_train,
        skip_header=False
    )
    field_test = field_maker('test')
    test_data = TabularDataset(
        path=test_file_name,
        format='csv',
        fields=field_test,
        skip_header=False
    )

    return train_data,test_data

def train_val_splitter(trn_data):
    train_data, val_data = trn_data.split(
        split_raio=0.8,
        random_state=random.seed(100)
    )
    return train_data, val_data

if __name__=="__main__":

    train_file = r'C:\Users\Ankan\Downloads\train_tweet.csv'
    test_file = r'C:\Users\Ankan\Downloads\test_tweets.csv'
    trn_data, tst_data = train_test_maker(train_file,test_file)
    print(vars(trn_data[0]))

