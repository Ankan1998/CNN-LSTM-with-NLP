# Takes pandas series
import re
import pandas as pd


def hastag_finder(sentence):
    return re.findall(r"#(\w+)", sentence)


def hastag_series(sentence_series):
    return sentence_series.apply(lambda x: hastag_finder(x))


def hastag_symbol_remover(sentence):
    return sentence.replace("#", "").replace("_", " ")


def hastag_remover(sentence, regex_list):
    for reg in regex_list:
        sentence = sentence.replace("#" + reg, "")
    return sentence


if __name__ == "__main__":
    csv_file = r'C:\Users\Ankan\Downloads\train_tweet.csv'
    df = pd.read_csv(csv_file)
    # ss = hastag_series(df['tweet'])
    # print(ss.head())
    # print(len(ss) == len(df))
    sentence = df['tweet'][0]
    # cleaned_sentence = hastag_remover(sentence,hastag_finder(sentence))
    rem_Sentence = hastag_symbol_remover(sentence)
    print(sentence)
    print(rem_Sentence)


