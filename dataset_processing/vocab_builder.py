from dataset_processing.train_test_data_maker import train_test_maker


def vocab_builder(train_data,max_vocab, min_freq):
    train_data.fields['text'].build_vocab(train_data, max_size=max_vocab,min_freq=min_freq)
    train_data.fields['labels'].build_vocab(train_data)
    return train_data

if __name__=="__main__":

    train_file = r'C:\Users\Ankan\Downloads\sub_train.csv'
    test_file = r'C:\Users\Ankan\Downloads\sub_test.csv'
    trn_data, tst_data = train_test_maker(train_file,test_file)
    txt_vocab,lbl_vocab = vocab_builder(trn_data,1000,2)
    print(txt_vocab.stoi)