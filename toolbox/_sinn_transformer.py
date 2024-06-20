#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn
import math

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :] 
    
#time Absolute Position Encoding (tAPE)
class tAPE(nn.Module): # Equation 13 page 11
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/max_len))
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)
    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)

    
# class RelativePosition(nn.Module):

#     def __init__(self, num_units, max_relative_position):
#         super().__init__()
#         self.num_units = num_units
#         self.max_relative_position = max_relative_position
#         self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
#         nn.init.xavier_uniform_(self.embeddings_table)

#     def forward(self, length_q, length_k):
#         range_vec_q = torch.arange(length_q)
#         range_vec_k = torch.arange(length_k)
#         distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
#         distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
#         final_mat = distance_mat_clipped + self.max_relative_position
#         final_mat = torch.LongTensor(final_mat).cuda()
#         embeddings = self.embeddings_table[final_mat].cuda()

#         return embeddings



# class PositionalEncoding(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, bidirectional=False):
#         super(PositionalEncoding, self).__init__()
#         self.lstm = nn.LSTM(input_size=input_size+1,  # Add 1 to input size for positional indices
#                             hidden_size=hidden_size,
#                             num_layers=num_layers,
#                             bidirectional=bidirectional,
#                             batch_first=True)
        
#     def forward(self, x):
#         seq_len, batch_size, input_size = x.size()
#         # Create positional indices
#         positions = torch.arange(seq_len).unsqueeze(1).repeat(1, batch_size).to(x.device)
#         # Concatenate positional indices with input
#         x = torch.cat((x, positions.unsqueeze(-1).float()), dim=-1)
#         # LSTM encoding
#         x, _ = self.lstm(x)
#         return x
    
      

class SINN_transformer(nn.Module):
    def __init__(self, input_dimension, d_model, nhead, encoder_num_layers, decoder_num_layers, output_dimension, dropout_p=0):
        super().__init__()
        self.embedding = nn.Linear(input_dimension, d_model)
#         self.pos_encoder = PositionalEncoding(input_dimension,d_model,num_layers=1)
        self.pos_encoder = PositionalEncoding(d_model)
#         self.pos_encoder = tAPE(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, encoder_num_layers)
        
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, decoder_num_layers)
        self.decoder = nn.Linear(d_model, output_dimension)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, ini, tgt):
        # encoder part
        x = self.embedding(ini)
        x = self.pos_encoder(x)
        encoder_output = self.transformer_encoder(x)

        # decoder part 
        y = self.embedding(tgt)
        y = self.pos_encoder(y)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(device=ini.device)
        y = self.transformer_decoder(y, encoder_output, tgt_mask=tgt_mask)
#         y = self.transformer_decoder(y, encoder_output)
        output = self.decoder(y)
        return output
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask