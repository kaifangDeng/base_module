import torch
import torch.nn as nn



class DynamicLSTM(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=True, dropout=0, bidirectional=True, device=torch.device('cpu')):
        """
        Dynamic LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size: The number of expected features in the input x
        :param hidden_size: The number of features in the hidden state h
        :param num_layers: Number of recurrent layers.
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout: If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional: If True, becomes a bidirectional RNN. Default: False
        """
        super(DynamicLSTM,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional  = bidirectional
        self.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.device = device
        self.to(device)

    def forward(self, x, sent_length, only_use_last_hidden_state=False):
        '''
        (1)当pack传入enforce_sorted==False时，pack函数会预先自动为input按照length进行排序，然后对排好序的input再进行pack；
        (2)如果pack时enforce_sorted==False，那么pad_packed_sequence的最终输出也会受到影响——在pack时排好序的input，在pad后会返回成原有的顺序；
        (3)即使pack时enforce_sorted==False，我们执行"LSTM(packed)"所返回出来的每个segment的features(即最后一个unit的hidden_state)，仍然是原始input的序列的。
        '''
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x,sent_length.cpu(),batch_first=self.batch_first,
                                                enforce_sorted=False)
        packed_out, (ht, ct) = self.LSTM(packed_x)

        # sent_length 和 length 相同
        output, length = torch.nn.utils.rnn.pad_packed_sequence(packed_out,batch_first=self.batch_first,
                                                                total_length=sent_length.max())
        # 是否只需要最后一个cell的hidden_state
        if only_use_last_hidden_state:
            return ht
        else:
            return output, (ht,ct)
