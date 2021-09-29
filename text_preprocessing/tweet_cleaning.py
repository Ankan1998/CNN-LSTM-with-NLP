import preprocessor as pr
import pandas as pd
import re

def cleanup_text(text):
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = re.sub(r' +', ' ', text)
    cleaned_text = re.sub(r'\n', ' ', text)

    return cleaned_text

def tweet_preprocessing_series(tweet_series):
    cleaned_list = []
    for v in tweet_series:
        init_clean = pr.clean(v)
        reg_clean = cleanup_text(init_clean)
        cleaned_list.append(reg_clean)
    return pd.Series(cleaned_list)

def single_tweet_preprocessing(tweet):
    clean_sentence = pr.clean(tweet)
    reg_sentence_clean = cleanup_text(clean_sentence)
    return reg_sentence_clean

if __name__=="__main__":
    csv_file = r'C:\Users\Ankan\Downloads\train_tweet.csv'
    df = pd.read_csv(csv_file)
    # df['cleaned'] = tweet_preprocessing_series(df['tweet'])
    print(df['tweet'][0])
    print(single_tweet_preprocessing(df['tweet'][0]))
    # print(df['cleaned'][0])