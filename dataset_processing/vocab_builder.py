from dataset_processing.custom_field import field_maker
from dataset_processing.train_test_data_maker import train_test_maker, train_val_splitter


def vocab_builder(train_data):
    train_data.fields['text'].build_vocab(train_data, max_size=100,min_freq=2)
    train_data.fields['labels'].build_vocab(train_data)
    return train_data

if __name__=="__main__":

    train_file = r'C:\Users\Ankan\Downloads\sub_train.csv'
    test_file = r'C:\Users\Ankan\Downloads\sub_test.csv'
    trn_data, tst_data = train_test_maker(train_file,test_file)
    txt_vocab,lbl_vocab = vocab_builder(trn_data)
    print(txt_vocab.stoi)