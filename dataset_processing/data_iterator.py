from torchtext.legacy.data import BucketIterator
def dataset_itr(train_data,val_data,test_data, BATCH_SIZE,device):
    train_itr, val_itr, test_itr = BucketIterator.splits(
        (train_data,val_data,test_data),
        batch_size= BATCH_SIZE,
        device = device,
        sort=False
    )

    return train_itr,val_itr,test_itr