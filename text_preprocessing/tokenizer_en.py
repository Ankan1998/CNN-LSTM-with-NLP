import spacy
import pandas as pd
from text_preprocessing.tweet_cleaning import tweet_preprocessing
import subprocess


try:
    en = spacy.load('en_core_web_sm')
except:
    _ = subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    en = spacy.load('en_core_web_sm')

def tokenize(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]

if __name__=="__main__":
    csv_file = r'C:\Users\Ankan\Downloads\train_tweet.csv'
    df = pd.read_csv(csv_file)
    df['cleaned'] = tweet_preprocessing(df['tweet'])
    print(df['cleaned'][0])
    print(tokenize(df['cleaned'][0]))
