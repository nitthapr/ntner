from modules.layers.decoders import *
from modules.layers.embedders import *
from modules.layers.layers import BiLSTM, MultiHeadAttention
import abc

#Add gensim from thai2fit word embeddings
from gensim.models import KeyedVectors
import torch
import torch.nn as nn


class BERTNerModel(nn.Module, metaclass=abc.ABCMeta):
    """Base class for all BERT Models"""

    @abc.abstractmethod
    def forward(self, batch):
        raise NotImplementedError("abstract method forward must be implemented")

    @abc.abstractmethod
    def score(self, batch):
        raise NotImplementedError("abstract method score must be implemented")

    @abc.abstractmethod
    def create(self, *args, **kwargs):
        raise NotImplementedError("abstract method create must be implemented")

    def get_n_trainable_params(self):
        pp = 0
        for p in list(self.parameters()):
            if p.requires_grad:
                num = 1
                for s in list(p.size()):
                    num = num * s
                pp += num
        return pp


class BERTBiLSTMCRF(BERTNerModel):

    def __init__(self, embeddings, lstm, crf, device="cuda"):
        super(BERTBiLSTMCRF, self).__init__()
        self.embeddings = embeddings
        self.lstm = lstm
        self.crf = crf
        self.to(device)

    def forward(self, batch):
        #input_, labels_mask, input_type_ids = batch[:3]
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        return self.crf.forward(output, labels_mask, labels)

    def score(self, batch):
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        return self.crf.score(output, labels_mask, labels)

    @classmethod
    def create(cls,
               label_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               # BiLSTM params
               embedding_size=768, hidden_dim=512, rnn_layers=1, lstm_dropout=0.3,
               # CRFDecoder params
               crf_dropout=0.5,
               # Global params
               device="cuda"):
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        lstm = BiLSTM.create(
                embedding_size=embedding_size, hidden_dim=hidden_dim, rnn_layers=rnn_layers, dropout=lstm_dropout)
        crf = CRFDecoder.create(label_size, hidden_dim, crf_dropout)
        return cls(embeddings, lstm, crf, device)


# additional by prapas: adding pos embeddings layer
class BERTPosBiLSTMCRF(BERTNerModel):

    def __init__(self, embeddings, posembeddings, lstm, crf, device="cuda"):
        super(BERTPosBiLSTMCRF, self).__init__()
        self.embeddings = embeddings
        self.posembeddings = posembeddings
        self.lstm = lstm
        self.crf = crf
        self.to(device)

    def forward(self, batch):
        #input_, labels_mask, input_type_ids = batch[:3]
        input_, labels_mask, input_type_ids, labels, pos_ids = batch
        input_embeddings = self.embeddings(batch)
        embedded_pos = self.posembeddings(pos_ids)
        output, _ = self.lstm.forward(torch.cat((input_embeddings,embedded_pos),2), labels_mask)
        return self.crf.forward(output, labels_mask, labels)

    def score(self, batch):
        input_, labels_mask, input_type_ids, labels, pos_ids = batch
        input_embeddings = self.embeddings(batch)
        embedded_pos = self.posembeddings(pos_ids)
        output, _ = self.lstm.forward(torch.cat((input_embeddings,embedded_pos),2), labels_mask)
        return self.crf.score(output, labels_mask, labels)

    @classmethod
    def create(cls,
               label_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               # Pos params
               num_pos_dict=18, pos_embedding_size=18,
               # BiLSTM params
               embedding_size=768, hidden_dim=512, rnn_layers=1, lstm_dropout=0.3,
               # CRFDecoder params
               crf_dropout=0.5,
               # Global params
               device="cuda"):
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        posembeddings = torch.nn.Embedding(num_pos_dict, pos_embedding_size)
        lstm = BiLSTM.create(
                embedding_size=embedding_size+pos_embedding_size, hidden_dim=hidden_dim, rnn_layers=rnn_layers, dropout=lstm_dropout)
        crf = CRFDecoder.create(label_size, hidden_dim, crf_dropout)
        return cls(embeddings, posembeddings, lstm, crf, device)


# additional by prapas: adding char embeddings layer
class BERTCharBiLSTMCRF(BERTNerModel):

    def __init__(self, embeddings, charencodings, lstm, crf, device="cuda"):
        super(BERTCharBiLSTMCRF, self).__init__()
        self.embeddings = embeddings
        self.charencodings = charencodings
        self.lstm = lstm
        self.crf = crf
        self.to(device)

    def forward(self, batch):
        #input_, labels_mask, input_type_ids = batch[:3]
        input_, labels_mask, input_type_ids, labels, char_ids = batch
        input_embeddings = self.embeddings(batch)
        char_enc = self.charencodings.forward(batch)
        output, _ = self.lstm.forward(torch.cat((input_embeddings,char_enc),2) , labels_mask)
        return self.crf.forward(output, labels_mask, labels)

    def score(self, batch):
        input_, labels_mask, input_type_ids, labels, char_ids = batch
        input_embeddings = self.embeddings(batch)
        char_enc = self.charencodings.forward(batch)
        output, _ = self.lstm.forward(torch.cat((input_embeddings,char_enc),2) , labels_mask)
        return self.crf.score(output, labels_mask, labels)

    @classmethod
    def create(cls,
               label_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               # Char Encoder params
               num_char_dict=399,
               # BiLSTM params
               embedding_size=768, hidden_dim=512, rnn_layers=1, lstm_dropout=0.5,
               # BiLSTM params for char encoders
               char_lstm_embedding_size=32, char_lstm_hidden_dim=32, char_lstm_rnn_layers=1, char_lstm_dropout=0.5,
               # CRFDecoder params
               crf_dropout=0.5,
               # Global params
               device="cuda"):
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        charencodings = CHAREncoder.create(num_char_dict=num_char_dict, embedding_size=char_lstm_embedding_size,hidden_dim=char_lstm_hidden_dim,rnn_layers=char_lstm_rnn_layers,lstm_dropout=char_lstm_dropout,device=device, mode=mode, is_freeze=is_freeze)
        lstm = BiLSTM.create(
                embedding_size=embedding_size+(char_lstm_hidden_dim*2), hidden_dim=hidden_dim, rnn_layers=rnn_layers, dropout=lstm_dropout)
        crf = CRFDecoder.create(label_size, hidden_dim, crf_dropout)
        return cls(embeddings, charencodings, lstm, crf, device)

# additional by prapas: adding char embeddings layer
class BERTCharBiLSTMCRF2(BERTNerModel):

    def __init__(self, embeddings, chart_lstm, lstm, crf, device="cuda"):
        super(BERTCharBiLSTMCRF2, self).__init__()
        self.embeddings = embeddings
        self.chart_lstm = chart_lstm
        self.lstm = lstm
        self.crf = crf
        self.device = device
        self.to(device)

    def forward(self, batch):
        #input_, labels_mask, input_type_ids = batch[:3]
        input_, labels_mask, input_type_ids, labels, char_ids = batch
        input_embeddings = self.embeddings(batch)
        char_enc, _ = self.chart_lstm.forward(char_ids.type(torch.FloatTensor).to(device=self.device))
        output, _ = self.lstm.forward(torch.cat((input_embeddings,char_enc),2) , labels_mask)
        return self.crf.forward(output, labels_mask, labels)

    def score(self, batch):
        input_, labels_mask, input_type_ids, labels, char_ids = batch
        input_embeddings = self.embeddings(batch)
        char_enc, _ = self.chart_lstm.forward(char_ids.type(torch.FloatTensor).to(device=self.device))
        output, _ = self.lstm.forward(torch.cat((input_embeddings,char_enc),2) , labels_mask)
        return self.crf.score(output, labels_mask, labels)

    @classmethod
    def create(cls,
               label_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               # Char Encoder params
               num_char_dict=189,
               max_token_char_len=30,
               # BiLSTM params
               embedding_size=768, hidden_dim=512, rnn_layers=1, lstm_dropout=0.5,
               # BiLSTM params for char encoders
               #char_lstm_embedding_size=5670, char_lstm_rnn_layers=1,
               char_lstm_hidden_dim=32, char_lstm_dropout=0.5,
               # CRFDecoder params
               crf_dropout=0.5,
               # Global params
               device="cuda"):
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        chart_lstm = torch.nn.LSTM(num_char_dict*max_token_char_len, char_lstm_hidden_dim, bidirectional=True, batch_first=True, dropout=char_lstm_dropout)
        lstm = BiLSTM.create(
                embedding_size=embedding_size+(char_lstm_hidden_dim*2), hidden_dim=hidden_dim, rnn_layers=rnn_layers, dropout=lstm_dropout)
        crf = CRFDecoder.create(label_size, hidden_dim, crf_dropout)
        return cls(embeddings, chart_lstm, lstm, crf, device)


class BERTBiLSTMNCRF(BERTNerModel):

    def __init__(self, embeddings, lstm, crf, device="cuda"):
        super(BERTBiLSTMNCRF, self).__init__()
        self.embeddings = embeddings
        self.lstm = lstm
        self.crf = crf
        self.to(device)

    def forward(self, batch):
        #input_, labels_mask, input_type_ids = batch[:3]
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        return self.crf.forward(output, labels_mask, labels)

    def score(self, batch):
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        return self.crf.score(output, labels_mask, labels)

    @classmethod
    def create(cls,
               label_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               # BiLSTM params
               embedding_size=768, hidden_dim=512, rnn_layers=1, lstm_dropout=0.3,
               # NCRFDecoder params
               crf_dropout=0.5, nbest=1,
               # Global params
               device="cuda"):
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        lstm = BiLSTM.create(
                embedding_size=embedding_size, hidden_dim=hidden_dim, rnn_layers=rnn_layers, dropout=lstm_dropout)
        crf = NCRFDecoder.create(
            label_size, hidden_dim, crf_dropout, nbest, device=device)
        return cls(embeddings, lstm, crf, device)


class BERTAttnCRF(BERTNerModel):

    def __init__(self, embeddings, attn, crf, device="cuda"):
        super(BERTAttnCRF, self).__init__()
        self.embeddings = embeddings
        self.attn = attn
        self.crf = crf
        self.to(device)

    def forward(self, batch):
        #input_, labels_mask, input_type_ids = batch[:3]
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.attn(input_embeddings, input_embeddings, input_embeddings, None)
        return self.crf.forward(output, labels_mask, labels)

    def score(self, batch):
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.attn(input_embeddings, input_embeddings, input_embeddings, None)
        return self.crf.score(output, labels_mask, labels)

    @classmethod
    def create(cls,
               label_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               # Attn params
               embedding_size=768, key_dim=64, val_dim=64, num_heads=3, attn_dropout=0.3,
               # CRFDecoder params
               crf_dropout=0.5,
               # Global params
               device="cuda"):
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        attn = MultiHeadAttention(key_dim, val_dim, embedding_size, num_heads, attn_dropout)
        crf = CRFDecoder.create(
            label_size, embedding_size, crf_dropout)
        return cls(embeddings, attn, crf, device)


class BERTAttnNCRF(BERTNerModel):

    def __init__(self, embeddings, attn, crf, device="cuda"):
        super(BERTAttnNCRF, self).__init__()
        self.embeddings = embeddings
        self.attn = attn
        self.crf = crf
        self.to(device)

    def forward(self, batch):
        #input_, labels_mask, input_type_ids = batch[:3]
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.attn(input_embeddings, input_embeddings, input_embeddings, None)
        return self.crf.forward(output, labels_mask, labels)

    def score(self, batch):
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.attn(input_embeddings, input_embeddings, input_embeddings, None)
        return self.crf.score(output, labels_mask, labels)

    @classmethod
    def create(cls,
               label_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               # Attn params
               embedding_size=768, key_dim=64, val_dim=64, num_heads=3, attn_dropout=0.3,
               # NCRFDecoder params
               crf_dropout=0.5, nbest=1,
               # Global params
               device="cuda"):
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        attn = MultiHeadAttention(key_dim, val_dim, embedding_size, num_heads, attn_dropout)
        crf = NCRFDecoder.create(
            label_size, embedding_size, crf_dropout, nbest=nbest, device=device)
        return cls(embeddings, attn, crf, device)


class BERTBiLSTMAttnCRF(BERTNerModel):

    def __init__(self, embeddings, lstm, attn, crf, device="cuda"):
        super(BERTBiLSTMAttnCRF, self).__init__()
        self.embeddings = embeddings
        self.lstm = lstm
        self.attn = attn
        self.crf = crf
        self.to(device)

    def forward(self, batch):
        #input_, labels_mask, input_type_ids = batch[:3]
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        output, _ = self.attn(output, output, output, None)
        return self.crf.forward(output, labels_mask, labels)

    def score(self, batch):
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        output, _ = self.attn(output, output, output, None)
        return self.crf.score(output, labels_mask, labels)

    @classmethod
    def create(cls,
               label_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               # BiLSTM
               hidden_dim=512, rnn_layers=1, lstm_dropout=0.3,
               # Attn params
               embedding_size=768, key_dim=64, val_dim=64, num_heads=3, attn_dropout=0.3,
               # CRFDecoder params
               crf_dropout=0.5,
               # Global params
               device="cuda"):
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        lstm = BiLSTM.create(
            embedding_size=embedding_size, hidden_dim=hidden_dim, rnn_layers=rnn_layers, dropout=lstm_dropout)
        attn = MultiHeadAttention(key_dim, val_dim, hidden_dim, num_heads, attn_dropout)
        crf = CRFDecoder.create(
            label_size, hidden_dim, crf_dropout)
        return cls(embeddings, lstm, attn, crf, device)


class BERTBiLSTMAttnNCRF(BERTNerModel):

    def __init__(self, embeddings, lstm, attn, crf, device="cuda"):
        super(BERTBiLSTMAttnNCRF, self).__init__()
        self.embeddings = embeddings
        self.lstm = lstm
        self.attn = attn
        self.crf = crf
        self.to(device)

    def forward(self, batch):
        #input_, labels_mask, input_type_ids = batch[:3]
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        output, _ = self.attn(output, output, output, None)
        return self.crf.forward(output, labels_mask, labels)

    def score(self, batch):
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        output, _ = self.attn(output, output, output, None)
        return self.crf.score(output, labels_mask, labels)

    @classmethod
    def create(cls,
               label_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               # BiLSTM
               hidden_dim=512, rnn_layers=1, lstm_dropout=0.3,
               # Attn params
               embedding_size=768, key_dim=64, val_dim=64, num_heads=3, attn_dropout=0.3,
               # NCRFDecoder params
               crf_dropout=0.5, nbest=1,
               # Global params
               device="cuda"):
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        lstm = BiLSTM.create(
            embedding_size=embedding_size, hidden_dim=hidden_dim, rnn_layers=rnn_layers, dropout=lstm_dropout)
        attn = MultiHeadAttention(key_dim, val_dim, hidden_dim, num_heads, attn_dropout)
        crf = NCRFDecoder.create(
            label_size, hidden_dim, crf_dropout, nbest=nbest, device=device)
        return cls(embeddings, lstm, attn, crf, device)


class BERTBiLSTMAttnNCRFJoint(BERTNerModel):

    def __init__(self, embeddings, lstm, attn, crf, clf, device="cuda"):
        super(BERTBiLSTMAttnNCRFJoint, self).__init__()
        self.embeddings = embeddings
        self.lstm = lstm
        self.attn = attn
        self.crf = crf
        self.clf = clf
        self.to(device)

    def forward(self, batch):
        #input_, labels_mask, input_type_ids = batch[:3]
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        output, _ = self.attn(output, output, output, None)
        return self.crf.forward(output, labels_mask, labels), self.clf(output)

    def score(self, batch):
        input_, labels_mask, input_type_ids, labels, cls_ids = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        output, _ = self.attn(output, output, output, None)
        return self.crf.score(output, labels_mask, labels) + self.clf.score(output, cls_ids)

    @classmethod
    def create(cls,
               label_size, intent_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               # BiLSTM
               hidden_dim=512, rnn_layers=1, lstm_dropout=0.3,
               # Attn params
               embedding_size=768, key_dim=64, val_dim=64, num_heads=3, attn_dropout=0.3,
               # NCRFDecoder params
               crf_dropout=0.5, nbest=1,
               # Clf params
               clf_dropout=0.3,
               # Global params
               device="cuda"):
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        lstm = BiLSTM.create(
            embedding_size=embedding_size, hidden_dim=hidden_dim, rnn_layers=rnn_layers, dropout=lstm_dropout)
        attn = MultiHeadAttention(key_dim, val_dim, hidden_dim, num_heads, attn_dropout)
        crf = NCRFDecoder.create(
            label_size, hidden_dim, crf_dropout, nbest=nbest, device=device)
        clf = ClassDecoder(intent_size, hidden_dim, clf_dropout)
        return cls(embeddings, lstm, attn, crf, clf, device)


class BERTBiLSTMAttnCRFJoint(BERTNerModel):

    def __init__(self, embeddings, lstm, attn, crf, clf, device="cuda"):
        super(BERTBiLSTMAttnCRFJoint, self).__init__()
        self.embeddings = embeddings
        self.lstm = lstm
        self.attn = attn
        self.crf = crf
        self.clf = clf
        self.to(device)

    def forward(self, batch):
        #input_, labels_mask, input_type_ids = batch[:3]
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        output, _ = self.attn(output, output, output, None)
        return self.crf.forward(output, labels_mask, labels), self.clf(output)

    def score(self, batch):
        input_, labels_mask, input_type_ids, labels, cls_ids = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        output, _ = self.attn(output, output, output, None)
        return self.crf.score(output, labels_mask, labels) + self.clf.score(output, cls_ids)

    @classmethod
    def create(cls,
               label_size, intent_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               # BiLSTM
               hidden_dim=512, rnn_layers=1, lstm_dropout=0.3,
               # Attn params
               embedding_size=768, key_dim=64, val_dim=64, num_heads=3, attn_dropout=0.3,
               # CRFDecoder params
               crf_dropout=0.5,
               # Clf params
               clf_dropout=0.3,
               # Global params
               device="cuda"):
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        lstm = BiLSTM.create(
            embedding_size=embedding_size, hidden_dim=hidden_dim, rnn_layers=rnn_layers, dropout=lstm_dropout)
        attn = MultiHeadAttention(key_dim, val_dim, hidden_dim, num_heads, attn_dropout)
        crf = CRFDecoder.create(
            label_size, hidden_dim, crf_dropout)
        clf = ClassDecoder(intent_size, hidden_dim, clf_dropout)
        return cls(embeddings, lstm, attn, crf, clf, device)


class BERTBiLSTMCRFJoint(BERTNerModel):

    def __init__(self, embeddings, lstm, crf, clf, device="cuda"):
        super(BERTBiLSTMCRFJoint, self).__init__()
        self.embeddings = embeddings
        self.lstm = lstm
        self.crf = crf
        self.clf = clf
        self.to(device)

    def forward(self, batch):
        #input_, labels_mask, input_type_ids = batch[:3]
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        return self.crf.forward(output, labels_mask, labels), self.clf(output)

    def score(self, batch):
        input_, labels_mask, input_type_ids, labels, cls_ids = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        return self.crf.score(output, labels_mask, labels) + self.clf.score(output, cls_ids)

    @classmethod
    def create(cls,
               label_size, intent_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               # BiLSTM params
               embedding_size=768, hidden_dim=512, rnn_layers=1, lstm_dropout=0.3,
               # CRFDecoder params
               crf_dropout=0.5,
               # Clf params
               clf_dropout=0.3,
               # Global params
               device="cuda"):
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        lstm = BiLSTM.create(
                embedding_size=embedding_size, hidden_dim=hidden_dim, rnn_layers=rnn_layers, dropout=lstm_dropout)
        crf = CRFDecoder.create(label_size, hidden_dim, crf_dropout)
        clf = ClassDecoder(intent_size, hidden_dim, clf_dropout)
        return cls(embeddings, lstm, crf, clf, device)


class BERTBiLSTMNCRFJoint(BERTNerModel):

    def __init__(self, embeddings, lstm, crf, clf, device="cuda"):
        super(BERTBiLSTMNCRFJoint, self).__init__()
        self.embeddings = embeddings
        self.lstm = lstm
        self.crf = crf
        self.clf = clf
        self.to(device)

    def forward(self, batch):
        #input_, labels_mask, input_type_ids = batch[:3]
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        return self.crf.forward(output, labels_mask, labels), self.clf(output)

    def score(self, batch):
        input_, labels_mask, input_type_ids, labels, cls_ids = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        return self.crf.score(output, labels_mask, labels) + self.clf.score(output, cls_ids)

    @classmethod
    def create(cls,
               label_size, intent_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               # BiLSTM params
               embedding_size=768, hidden_dim=512, rnn_layers=1, lstm_dropout=0.3,
               # CRFDecoder params
               crf_dropout=0.5, nbest=1,
               # Clf params
               clf_dropout=0.3,
               # Global params
               device="cuda"):
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        lstm = BiLSTM.create(
                embedding_size=embedding_size, hidden_dim=hidden_dim, rnn_layers=rnn_layers, dropout=lstm_dropout)
        crf = NCRFDecoder.create(label_size, hidden_dim, crf_dropout, nbest=nbest, device=device)
        clf = ClassDecoder(intent_size, hidden_dim, clf_dropout)
        return cls(embeddings, lstm, crf, clf, device)


class BERTAttnCRFJoint(BERTNerModel):

    def __init__(self, embeddings, attn, crf, clf, device="cuda"):
        super(BERTAttnCRFJoint, self).__init__()
        self.embeddings = embeddings
        self.attn = attn
        self.crf = crf
        self.clf = clf
        self.to(device)

    def forward(self, batch):
        #input_, labels_mask, input_type_ids = batch[:3]
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.attn(input_embeddings, input_embeddings, input_embeddings, None)
        return self.crf.forward(output, labels_mask, labels), self.clf(output)

    def score(self, batch):
        input_, labels_mask, input_type_ids, labels, cls_ids = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.attn(input_embeddings, input_embeddings, input_embeddings, None)
        return self.crf.score(output, labels_mask, labels) + self.clf.score(output, cls_ids)

    @classmethod
    def create(cls,
               label_size, intent_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               # BiLSTM
               hidden_dim=512, rnn_layers=1, lstm_dropout=0.3,
               # Attn params
               embedding_size=768, key_dim=64, val_dim=64, num_heads=3, attn_dropout=0.3,
               # CRFDecoder params
               crf_dropout=0.5,
               # Clf params
               clf_dropout=0.3,
               # Global params
               device="cuda"):
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        attn = MultiHeadAttention(key_dim, val_dim, hidden_dim, num_heads, attn_dropout)
        crf = CRFDecoder.create(
            label_size, hidden_dim, crf_dropout)
        clf = ClassDecoder(intent_size, hidden_dim, clf_dropout)
        return cls(embeddings, attn, crf, clf, device)


class BERTAttnNCRFJoint(BERTNerModel):

    def __init__(self, embeddings, attn, crf, clf, device="cuda"):
        super(BERTAttnNCRFJoint, self).__init__()
        self.embeddings = embeddings
        self.attn = attn
        self.crf = crf
        self.clf = clf
        self.to(device)

    def forward(self, batch):
        #input_, labels_mask, input_type_ids = batch[:3]
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.attn(input_embeddings, input_embeddings, input_embeddings, None)
        return self.crf.forward(output, labels_mask, labels), self.clf(output)

    def score(self, batch):
        input_, labels_mask, input_type_ids, labels, cls_ids = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.attn(input_embeddings, input_embeddings, input_embeddings, None)
        return self.crf.score(output, labels_mask, labels) + self.clf.score(output, cls_ids)

    @classmethod
    def create(cls,
               label_size, intent_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               # BiLSTM
               hidden_dim=512, rnn_layers=1, lstm_dropout=0.3,
               # Attn params
               embedding_size=768, key_dim=64, val_dim=64, num_heads=3, attn_dropout=0.3,
               # NCRFDecoder params
               crf_dropout=0.5, nbest=1,
               # Clf params
               clf_dropout=0.3,
               # Global params
               device="cuda"):
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        attn = MultiHeadAttention(key_dim, val_dim, hidden_dim, num_heads, attn_dropout)
        crf = NCRFDecoder.create(
            label_size, hidden_dim, crf_dropout, nbest=nbest, device=device)
        clf = ClassDecoder(intent_size, hidden_dim, clf_dropout)
        return cls(embeddings, attn, crf, clf, device)


class BERTNCRF(BERTNerModel):

    def __init__(self, embeddings, crf, device="cuda"):
        super(BERTNCRF, self).__init__()
        self.embeddings = embeddings
        self.crf = crf
        self.to(device)

    def forward(self, batch):
        #input_, labels_mask, input_type_ids = batch[:3]
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        return self.crf.forward(input_embeddings, labels_mask, labels)

    def score(self, batch):
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        return self.crf.score(input_embeddings, labels_mask, labels)

    @classmethod
    def create(cls,
               label_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               embedding_size=768,
               # NCRFDecoder params
               crf_dropout=0.5, nbest=1,
               # Global params
               device="cuda"):
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        crf = NCRFDecoder.create(
            label_size, embedding_size, crf_dropout, nbest=nbest, device=device)
        return cls(embeddings, crf, device)


class BERTCRF(BERTNerModel):

    def __init__(self, embeddings, crf, device="cuda"):
        super(BERTCRF, self).__init__()
        self.embeddings = embeddings
        self.crf = crf
        self.to(device)

    def forward(self, batch):
        #input_, labels_mask, input_type_ids = batch[:3]
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        return self.crf.forward(input_embeddings, labels_mask, labels)

    def score(self, batch):
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        return self.crf.score(input_embeddings, labels_mask, labels)

    @classmethod
    def create(cls,
               label_size,
               # BertEmbedder params
               model_name='bert-base-multilingual-cased', mode="weighted", is_freeze=True,
               embedding_size=768,
               # NCRFDecoder params
               crf_dropout=0.5,
               # Global params
               device="cuda"):
        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        crf = CRFDecoder.create(
            label_size, embedding_size, crf_dropout)
        return cls(embeddings, crf, device)
