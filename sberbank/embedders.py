from pytorch_pretrained_bert import BertModel
import torch

class BERTEmbedder(torch.nn.Module):
    def __init__(self, model, config):
        super(BERTEmbedder, self).__init__()
        self.config = config
        self.model = model
        if self.config["mode"] == "weighted":
            self.bert_weights = torch.nn.Parameter(torch.FloatTensor(12, 1))
            self.bert_gamma = torch.nn.Parameter(torch.FloatTensor(1, 1))
        self.init_weights()

    def init_weights(self):
        if self.config["mode"] == "weighted":
            torch.nn.init.xavier_normal(self.bert_gamma)
            torch.nn.init.xavier_normal(self.bert_weights)

    @classmethod
    def create(
            cls, model_name='bert-base-multilingual-cased',
            device="cuda", mode="weighted",
            is_freeze=True):
        config = {
            "model_name": model_name,
            "device": device,
            "mode": mode,
            "is_freeze": is_freeze
        }
        ### customize by prapas : switch to loading model from ThaiKeras-bert
        if model_name == 'bert-th':
            # bert_th_pytorch_dir: path where store bert pytorch model with bert config file
            bert_th_pytorch_dir='/content/drive/My Drive/Colab Notebooks/IS_NER/data/03_BERT_Thai_NER/wk'
            model = BertModel.from_pretrained(bert_th_pytorch_dir)
        else:
            model = BertModel.from_pretrained(model_name)
        ### end - customize by prapas
        model.to(device)
        model.train()
        self = cls(model, config)
        if is_freeze:
            self.freeze()
        return self

    @classmethod
    def from_config(cls, config):
        return cls.create(**config)

    def forward(self, batch):
        """
        batch has the following structure:
            data[0]: list, tokens ids
            data[1]: list, tokens mask
            data[2]: list, tokens type ids (for bert)
            data[3]: list, bert labels ids
        """
        encoded_layers, _ = self.model(
            input_ids=batch[0],
            token_type_ids=batch[2],
            attention_mask=batch[1],
            output_all_encoded_layers=self.config["mode"] == "weighted")
        if self.config["mode"] == "weighted":
            encoded_layers = torch.stack([a * b for a, b in zip(encoded_layers, self.bert_weights)])
            return self.bert_gamma * torch.sum(encoded_layers, dim=0)
        return encoded_layers

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False


# must implement character embedding + LSTM layer
class CHAREncoder(torch.nn.Module):
    def __init__(self, charembeddings, lstm, config):
        super(CHAREncoder, self).__init__()
        self.charembeddings = charembeddings
        self.lstm = lstm
        self.config = config
        if self.config["mode"] == "weighted":
            self.bert_weights = torch.nn.Parameter(torch.FloatTensor(12, 1))
            self.bert_gamma = torch.nn.Parameter(torch.FloatTensor(1, 1))
        self.init_weights()

    def init_weights(self):
        if self.config["mode"] == "weighted":
            torch.nn.init.xavier_normal(self.bert_gamma)
            torch.nn.init.xavier_normal(self.bert_weights)

    @classmethod
    def create(
            cls, 
            # Embedding params
            num_char_dict=399,
            # BiLSTM params
            embedding_size=32, hidden_dim=512, rnn_layers=1, lstm_dropout=0.3,
            device="cuda", mode="weighted",
            is_freeze=True):
        config = {
            "embedding_size": embedding_size,
            "hidden_dim": hidden_dim,
            "rnn_layers": rnn_layers,
            "lstm_dropout": lstm_dropout,
            "mode": mode,
            "is_freeze": is_freeze
        }
        charembeddings = torch.nn.Embedding(num_char_dict, embedding_size)
        lstm = torch.nn.LSTM(embedding_size, hidden_dim, bidirectional=True, batch_first=True)
        self = cls(charembeddings, lstm, config)
        return self

    @classmethod
    def from_config(cls, config):
        return cls.create(**config)

    def forward(self, batch):
        char_ids = batch[4]
        embedered_char = self.charembeddings(char_ids)
        embedered_char = torch.stack([a * b for a, b in zip(embedered_char, self.bert_weights)])
        embedered_char2 = self.bert_gamma * torch.sum(embedered_char, dim=0)
        #label_mask = batch[3]
        #length = label_mask.sum(-1)
        length = torch.tensor([30])
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        embedered_char2 = embedered_char2[sorted_idx]
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embedered_char2, sorted_lengths.data.tolist(), batch_first=True)
        output, (hidden, _) = self.lstm(packed_input) #The input dimension need to be (batch_size, seq_len, input_size)
        padded_outputs = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]
        _, reversed_idx = torch.sort(sorted_idx)
        return padded_outputs[reversed_idx], hidden[:, reversed_idx]

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False