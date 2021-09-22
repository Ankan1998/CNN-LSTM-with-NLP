import pandas as pd
from torchtext.legacy.data import Field
from text_preprocessing import tweet_cleaning
from text_preprocessing import tokenizer_en

def field_maker():
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
        unk_token=None
    )

    fields = [('text', Text), ('labels', Label)]
    return fields

if __name__=="__main__":
    csv_file = r'C:\Users\Ankan\Downloads\train_tweet.csv'
    df = pd.read_csv(csv_file)