import torch.cuda
from dataset_processing.dataset_processing_pipeline import data_pipeline
import torch.nn as nn
import torch.optim as optim
import json

from model.model import CNNNLPModel
from tqdm import tqdm

from saving_loading_model.saving import save_checkpoint
from training.train import train
from training.evaluate import evaluate

def main(
        train_file,
        test_file,
        config_file,
        checkpoint_path,
        best_model_path
    ):
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

    train_itr, val_itr, test_itr, vocab_size = data_pipeline(
        train_file,
        test_file,
        config['training']['max_vocab'],
        config['training']['min_freq'],
        config['training']['batch_size']
    )

    model = CNNNLPModel(
        vocab_size,
        config['model']['emb_dim'],
        config['model']['hid_dim'],
        config['model']['model_layer'],
        config['model']['model_kernel_size'],
        config['model']['model_dropout'],
        device
    )
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    num_epochs = config['training']['n_epoch']
    clip = config['training']['clip']
    is_best = False
    best_valid_loss = float('inf')

    for epoch in tqdm(range(num_epochs)):

        train_loss = train(model, train_itr, optimizer, criterion, clip)
        valid_loss = evaluate(model, val_itr, criterion)

        if (epoch + 1) % 2 == 0:
            print("training loss {}, validation_loss{}".format(train_loss,valid_loss))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            is_best = True
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, is_best, checkpoint_path, best_model_path)

if __name__=="__main__":
    train_file = r'C:\Users\Ankan\Downloads\train_tweet.csv'
    test_file = r'C:\Users\Ankan\Downloads\test_tweets.csv'
    config_file = r'C:\Users\Ankan\Desktop\Github\CNN-with-NLP\config.json'
    ckpt_path = r'C:\Users\Ankan\Desktop\Github\CNN-with-NLP\ckpt_best_path\latest.pt'
    best_model_path = r'C:\Users\Ankan\Desktop\Github\CNN-with-NLP\ckpt_best_path\best.pt'
    main(train_file,test_file,config_file,ckpt_path,best_model_path)