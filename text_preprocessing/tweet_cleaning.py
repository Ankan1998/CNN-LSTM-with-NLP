import preprocessor as pr
import pandas as pd

def tweet_preprocessing(tweet_series):
    cleaned_list = []
    for v in tweet_series:
        cleaned_list.append(pr.clean(v))
    return pd.Series(cleaned_list)

if __name__=="__main__":
    csv_file = r'C:\Users\Ankan\Downloads\train_tweet.csv'
    df = pd.read_csv(csv_file)
    df['cleaned'] = tweet_preprocessing(df['tweet'])
    print(df['tweet'][0])
    print(df['cleaned'][0])