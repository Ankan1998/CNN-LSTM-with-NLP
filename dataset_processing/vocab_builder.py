from dataset_processing.train_test_data_maker import train_test_maker
from helper.pickler import Objectpickler

def vocab_builder(train_data,max_vocab, min_freq):
    train_data.fields['text'].build_vocab(train_data, max_size=max_vocab,min_freq=min_freq)
    train_data.fields['labels'].build_vocab(train_data)
    return train_data


def vocab_pickler(train_data, pkl_filename):
    vocab_dict = {}
    vocab_dict['word'] = {}
    vocab_dict['labels'] = {}
    vocab_dict['word']['stoi'] = train_data.fields['text'].vocab.stoi
    vocab_dict['word']['itos'] = train_data.fields['text'].vocab.itos
    vocab_dict['labels']['stoi'] = train_data.fields['labels'].vocab.stoi
    vocab_dict['labels']['itos'] = train_data.fields['labels'].vocab.itos
    print(vocab_dict)
    Objectpickler(pkl_filename,vocab_dict)


if __name__=="__main__":

    train_file = r'C:\Users\Ankan\Downloads\sub_train.csv'
    test_file = r'C:\Users\Ankan\Downloads\sub_test.csv'
    trn_data, tst_data = train_test_maker(train_file,test_file)
    trn_vocab = vocab_builder(trn_data,1000,1)
    print(trn_vocab.stoi)