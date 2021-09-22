from text_preprocessing.tweet_cleaning import tweet_preprocessing
import pandas as pd

def max_doc_len(sentence_series):
    return sentence_series.map(len).max()

if __name__=="__main__":
    csv_file = r'C:\Users\Ankan\Downloads\train_tweet.csv'
    df = pd.read_csv(csv_file)
    df['cleaned'] = tweet_preprocessing(df['tweet'])
    print(max_doc_len(df['cleaned']))