from unicodedata import bidirectional
import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel


class M_BERT_LSTM(nn.module):
    def __init__(self, config, device):
        super(M_BERT_LSTM, self).__init__()
        self.config = config
        self.input_dim = 768
        self.hidden_dim = 256
        self.seq_len = 128 # need to check
        self.num_layers = 1
        self.checkpoint = 'bert-base-multilingual-cased'
        self.lstm_init_values = None

    def build_model(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.checkpoint)
        self.mbert_encoder = BertModel.from_pretrained(self.checkpoint)
        self.bi_lstm_encoder = nn.LSTM(input_size=self.input_dim, 
                                       hidden_size=self.hidden_dim,
                                       num_layers=self.num_layers,
                                       batch_first=True,
                                       bidirectional=True)
        
        self.lstm_decoder = nn.LSTM(input_size=self.hidden_dim,
                                    hidden_size=self.hidden_dim,
                                    num_layers=self.num_layers,
                                    batch_first=True,
                                    bidirectional=False)
        
    def forward(self, batch):
        question_list, equation_list = batch
        tokenized_questions = self.tokenizer(question_list,
                                             padding=True,
                                             truncation=True,
                                             return_tensors='pt')
        
        
        # forward encoder
        mbert_encoder_output = self.mbert_encoder(**tokenized_questions)[0]  # last hidden states (batch_size, sequence_length, hidden_size)
        bi_lstm_encoder_output = self.bi_lstm_encoder(mbert_encoder_output)
        
        # forward decoder
        lstm_decoder_output = self.lstm_decoder(bi_lstm_encoder_output)
        
        return lstm_decoder_output