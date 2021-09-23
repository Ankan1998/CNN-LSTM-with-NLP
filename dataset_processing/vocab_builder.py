from dataset_processing.custom_field import field_maker


def vocab_builder(train_data,val_data):
    field = field_maker()
    text = field['tweet'][1]
    text.build_vocab(train_data,val_data, max_size=10000)
    return text.vocab


