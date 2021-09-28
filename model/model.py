import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_processing.dataset_processing_pipeline import data_pipeline


class CNNNLPModel(nn.Module):
    def __init__(self,
                 input_dim,
                 emb_dim,
                 hid_dim,
                 n_layers,
                 kernel_size,
                 dropout,
                 device,
                 max_length=100):
        super().__init__()
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                              out_channels=2 * hid_dim,
                                              kernel_size=kernel_size,
                                              padding=(kernel_size - 1) // 2)
                                    for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)
        self.last_linear = nn.Linear(2*emb_dim,1)

    def forward(self, src):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        tok_embedded = self.tok_embedding(src)
        pos_embedded = self.pos_embedding(pos)
        embedded = self.dropout(tok_embedded + pos_embedded)
        conv_input = self.emb2hid(embedded)
        conv_input = conv_input.permute(0, 2, 1)

        for i, conv in enumerate(self.convs):
            conved = conv(self.dropout(conv_input))
            conved = F.glu(conved, dim=1)
            conved = (conved + conv_input) * self.scale
            conv_input = conved

        conved = self.hid2emb(conved.permute(0, 2, 1))
        combined = (conved + embedded) * self.scale


        combo = torch.cat((conved,combined),2)
        out = self.last_linear(combo)
        return out

if __name__=="__main__":

    train_file = r'C:\Users\Ankan\Downloads\sub_train.csv'
    test_file = r'C:\Users\Ankan\Downloads\sub_test.csv'
    train_itr, val_itr, test_itr = data_pipeline(train_file,test_file,2)
    INPUT_DIM = 100
    EMB_DIM = 256
    HID_DIM = 512  # each conv. layer has 2 * hid_dim filters
    ENC_LAYERS = 10  # number of conv. blocks in encoder
    ENC_KERNEL_SIZE = 3  # must be odd!
    ENC_DROPOUT = 0.25
    enc = CNNNLPModel(INPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT, 'cpu')
    for batch in train_itr:
        output = enc(batch.text)
        # print(batch.labels)
        # print(batch.fields['labels'])
        break