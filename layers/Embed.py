import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    '''
    pe(pos, 2i) = sin(w * pos)
    pe(pos, 2i+1) = cos(w * pos)
    w = 1/10000^(2i/d_model)
    to enable the numerical stability, w is rewritten as e^(ln(1/10000^(2i/d_model))),
    i.e. w = e^(-ln(10000^(2i/d_model))) = e^(-2i/d_model*ln(10000))
    '''
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model).float()
        # positional encoding are constants and do not require updating parameters
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        # the form of the frequency formula is different from that in the paper to avoid dividing 1
        # by a number that is close to 0, which can improve the numerical stability.
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # broadcasting mechanism
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # consider the batch_size dimension
        pe = pe.unsqueeze(0)
        # save the pe matrix into the state_dict(this will register a buffer for pe matrix,
        # to ensure the matrix can be saved as model parameters)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x.size(1): the position number(sequence length of the input)
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    """
    Map tokens to match the dimensions of the model
    """
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, 
                                   stride=1, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            
    def forward(self, x):
        # x: [B, S, C_in] -> [B, C_in, S] -> [B, D, S] -> [B, S, D]
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(2, 1)
        return x
    
class FixedEmbedding(nn.Module):
    """
    Map the time feature to match the dimensions of the model using positional encoding
    """
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(c_in, d_model)
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -math.log(10000.0) / d_model).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.embed = nn.Embedding(c_in, d_model)
        self.embed.weight = nn.parameter(w, required_grad = False)

    def forward(self, x):
        x = self.embed(x).detach()
        return x
    
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        # the time scale of minute is 15m, therefore the size is 4
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        
        # Three kinds of timefeature embedding: 
        # 1: FixedEmbedding(using positonal encoding)
        # 2: nn.Embedding(using the default embedding)
        # 3: TimefeatureEmbedding(using linear projection)
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        # use torch.long() to transfer datatype to torch.int64, ensuring the datatype of 
        # the input x [B, S, 5] are int and can be used as a index to get the embedding vector 
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.day_embed(x[:, :, 0])
        return minute_x + hour_x + weekday_x + day_x + month_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='TimeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        """
        The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly

        > `time_features` takes in a `dates` dataframe with a 'dates' column and extracts the date down to `freq` where freq can be any of the following if `timeenc` is 0:
        > * m - [month]
        > * w - [month]
        > * d - [month, day, weekday]
        > * b - [month, day, weekday]
        > * h - [month, day, weekday, hour]
        > * t - [month, day, weekday, hour, *minute]
        >
        > If `timeenc` is 1, a similar, but different list of `freq` values are supported (all encoded between [-0.5 and 0.5]):
        > * Q - [month]
        > * M - [month]
        > * W - [Day of month, week of year]
        > * D - [Day of week, day of month, day of year]
        > * B - [Day of week, day of month, day of year]
        > * H - [Hour of day, day of week, day of month, day of year]
        > * T - [Minute of hour*, hour of day, day of week, day of month, day of year]
        > * S - [Second of minute, minute of hour, hour of day, day of week, day of month, day of year]
        *minute returns a number from 0-3 corresponding to the 15 minute period it falls into.
        More details please refer to utils.timefeatures.py
        """
        freq_map = {'a': 1, 'm': 1, 'w': 2, 'd': 3,
                    'b': 3, 'h': 4, 's': 6, 't': 5}
        d_input = freq_map[freq]
        self.embed = nn.Linear(d_input, d_model, bias=False)

    def forward(self, x):            
        return self.embed(x)

# add position embedding and temporal_embedding(optional) to value embedding
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, 
                                                       embed_type=embed_type,
                                                        freq=freq) if embed_type != 'TimeF' else TimeFeatureEmbedding(d_model=d_model
                                                                                                                      , embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x)\
                    + self.position_embedding(x)
        return self.dropout(x)
    
    