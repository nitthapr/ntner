from pytorch_pretrained_bert import BertModel
import torch
from modules.layers.layers import BiLSTM

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


# implement character embedding with nn.embedding + BiLSTM layer with nn.LSTM
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

    def _to_one_hot(self, y, num_classes):
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype).to(device=self.config["device"])
        return zeros.scatter(scatter_dim, y_tensor, 1)

    @classmethod
    def create(
            cls, 
            # Embedding params
            num_char_dict=399,
            # BiLSTM params
            embedding_size=32, hidden_dim=32, rnn_layers=1, lstm_dropout=0.5,
            device="cuda", mode="weighted",
            is_freeze=True):
        config = {
            "embedding_size": embedding_size,
            "num_char_dict": num_char_dict,
            "hidden_dim": hidden_dim,
            "rnn_layers": rnn_layers,
            "lstm_dropout": lstm_dropout,
            "device": device,
            "mode": mode,
            "is_freeze": is_freeze
        }
        charembeddings = torch.nn.Embedding(num_char_dict, embedding_size)
        lstm = torch.nn.LSTM(embedding_size, hidden_dim, bidirectional=True, batch_first=True, dropout=lstm_dropout)
        self = cls(charembeddings, lstm, config)
        return self

    @classmethod
    def from_config(cls, config):
        return cls.create(**config)

    def forward(self, batch):
        #Alternative1: using method look like sberbank for word embedding
        #char_ids = batch[4]
        #embedered_char = self.charembeddings(char_ids)
        #embedered_char2 = self.bert_gamma * torch.sum(embedered_char, dim=2)
        #output, (hidden, _) = self.lstm(embedered_char2) #The input dimension need to be (batch_size, seq_len, input_size)        
        #return output

        #Alternative2: using forward method same as this example https://discuss.pytorch.org/t/implementation-character-embeddings-for-bilstm-ner/91976
        char_ids = batch[4]
        embedered_char = self.charembeddings(char_ids)
        char_vec = torch.empty(embedered_char.shape[0], embedered_char.shape[1], self.config["hidden_dim"]*2).to(device=self.config["device"])
        for idx, ch in enumerate(embedered_char):
            s_ch_rep, _ = self.lstm(ch)
            s_ch_rep_f = s_ch_rep[:, -1, 0: self.config["embedding_size"]]
            s_ch_rep_b = s_ch_rep[:, 0, self.config["embedding_size"]:]
            s_ch_rep = torch.cat((s_ch_rep_f, s_ch_rep_b), dim=1)
            char_vec[idx] = s_ch_rep
        return char_vec

        #Alternative3: make one hot before input to char embedding
        #char_ids = batch[4]
        #res_arr = []
        #res_shape = torch.LongTensor(char_ids.shape[0], char_ids.shape[1], char_ids.shape[2], self.config["num_char_dict"]).to(device=self.config["device"])
        #for rc in char_ids:
        #    arr = []
        #    r_shape = torch.LongTensor(char_ids.shape[1], char_ids.shape[2], self.config["num_char_dict"]).to(device=self.config["device"])
        #    for char_seq in rc:
        #        x = self._to_one_hot(y=char_seq, num_classes=self.config["num_char_dict"])
        #        arr.append(x.unsqueeze(0))
        #    r = torch.cat(arr, out=r_shape)
        #    res_arr.append(r.unsqueeze(0))
        #char_one_hot = torch.cat(res_arr, out=res_shape)
        #embedered_char = self.charembeddings(char_one_hot)        
        #char_vec = torch.empty(embedered_char.shape[0], embedered_char.shape[1], embedered_char.shape[4]*2).to(device=self.config["device"])
        #for idx, ch in enumerate(embedered_char):
        #    embedered_char2 = self.bert_gamma * torch.sum(ch, dim=2)
        #    s_ch_rep, _ = self.lstm(embedered_char2)
        #    s_ch_rep_f = s_ch_rep[:, -1, 0: self.config["embedding_size"]]
        #    s_ch_rep_b = s_ch_rep[:, 0, self.config["embedding_size"]:]
        #    s_ch_rep = torch.cat((s_ch_rep_f, s_ch_rep_b), dim=1)
        #    char_vec[idx] = s_ch_rep
        #return char_vec
        
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

# implement character embedding with nn.embedding + BiLSTM layer from sberbank code
class CHAREncoder2(torch.nn.Module):
    def __init__(self, charembeddings, lstm, config):
        super(CHAREncoder2, self).__init__()
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
            embedding_size=32, hidden_dim=64, rnn_layers=1, lstm_dropout=0.5,
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
        lstm = BiLSTM.create(
                embedding_size=embedding_size, hidden_dim=hidden_dim, rnn_layers=rnn_layers, dropout=lstm_dropout)
        self = cls(charembeddings, lstm, config)
        return self

    @classmethod
    def from_config(cls, config):
        return cls.create(**config)

    def forward(self, batch):
        char_ids = batch[4]
        embedered_char = self.charembeddings(char_ids)
        embedered_char2 = self.bert_gamma * torch.sum(embedered_char, dim=2)
        output, (hidden, _) = self.lstm(embedered_char2, batch[1])
        return output

        
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False