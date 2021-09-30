import torch
import torch.nn as nn
import pandas as pd
import json
from saving_loading_model.loading import load_for_inference
from model.model import CNNNLPModel
from text_preprocessing.tweet_cleaning import tweet_preprocessing_series,single_tweet_preprocessing
from text_preprocessing.tokenizer_en import tokenize
from helper.pickler import pickleloader

def inference(input_sentence,model_path,vocab_path):

    config_file = r'C:\Users\Ankan\Desktop\Github\CNN-with-NLP\config.json'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(config_file, 'r') as j:
        config = json.loads(j.read())

    for k,v in config['model'].items():
        v = float(v)
        if v < 1.0:
            config['model'][k] = float(v)
        else:
            config['model'][k] = int(v)

    for k,v in config['training'].items():
        v = float(v)
        if v < 1.0:
            config['training'][k] = float(v)
        else:
            config['training'][k] = int(v)

    model = CNNNLPModel(
        config['training']['max_vocab']+2,
        config['model']['emb_dim'],
        config['model']['hid_dim'],
        config['model']['model_layer'],
        config['model']['model_kernel_size'],
        config['model']['model_dropout'],
        device
    )

    predict_model = load_for_inference(model_path,model)
    clean_sentence = single_tweet_preprocessing(input_sentence)
    tokenized_sentence = tokenize(clean_sentence)
    vocab_dict = pickleloader(vocab_path)
    numericalize = [vocab_dict['word']['stoi'][val] for val in tokenized_sentence]
    tensor_input = torch.LongTensor(numericalize).unsqueeze(0).to(device)
    predict_model.eval()
    with torch.no_grad():
        output = predict_model(tensor_input)
    result = torch.round(torch.sigmoid(output.squeeze(0)))
    return result.item()

if __name__=="__main__":
    csv_file = r'C:\Users\Ankan\Downloads\train_tweet.csv'
    df = pd.read_csv(csv_file)
    model_path = r'C:\Users\Ankan\Desktop\Github\CNN-with-NLP\ckpt_best_path\best.pt'
    vocab_path = r'C:\Users\Ankan\Desktop\Github\CNN-with-NLP\vocab_pickler.pkl'
    res = inference(df['tweet'][77],model_path,vocab_path)
    print(res)



