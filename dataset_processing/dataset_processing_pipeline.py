from dataset_processing.train_test_data_maker import train_test_maker, train_val_splitter
from dataset_processing.vocab_builder import vocab_builder
from dataset_processing.data_iterator import dataset_itr

def data_pipeline(train_file, test_file,batch_size=128):

    # Cleaned Data
    trn_data, test_data = train_test_maker(train_file,test_file)

    # Build Vocab
    train_data = vocab_builder(trn_data)

    # Train and Val Split
    train_data, val_data = train_val_splitter(train_data)

    # Iterator
    train_itr, val_itr, test_itr = dataset_itr(train_data, val_data, test_data,batch_size)

    return train_itr, val_itr, test_itr

if __name__=="__main__":

    train_file = r'C:\Users\Ankan\Downloads\sub_train.csv'
    test_file = r'C:\Users\Ankan\Downloads\sub_test.csv'
    train_itr, val_itr, test_itr = data_pipeline(train_file,test_file,2)
    for batch in train_itr:
        print(batch.text)
        # print(batch.labels)
        # print(batch.fields['labels'])
        break

