import pandas as pd
from torchtext.legacy.data import Field
from text_preprocessing import tweet_cleaning
from text_preprocessing import tokenizer_en

def field_maker(type='train'):
    Text = Field(
        preprocessing=tweet_cleaning.tweet_preprocessing,
        sequential=True,
        tokenize=tokenizer_en.tokenize,
        batch_first=True,
        lower=True
    )
    Label = Field(
        sequential=False,
        use_vocab=False,
        pad_token=None,
        unk_token=None,
        batch_first=True
    )
    if type == 'train':
        fields = {'tweet':('text', Text), 'label': ('labels', Label)}
    else:
        fields ={'tweet':('text', Text)}
    return fields
