import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GCNLayer import XavierLinear

class AttentionLayer(nn.Module):
    def __init__(self, D, H, return_sequences=False):
        '''
        A single convolutional unit
        :param D: int, input feature dim
        :param H: int, hidden feature dim
        :param return_sequences: boolean, whether return sequence
        '''
        super(AttentionLayer, self).__init__()

        # Config copying
        self.H = H
        self.return_sequences = return_sequences
        self.D = D
        # self.linear1 = XavierLinear(D, H)
        # self.linear2 = XavierLinear(H, 1)
        self.linear_W = XavierLinear(H, H)

    def softmax_mask(self, x, mask):
        '''
        Softmax with mask
        :param x: torch.FloatTensor, logits, [batch_size, seq_len]
        :param mask: torch.ByteTensor, masks for sentences, [batch_size, seq_len]
        :return: torch.FloatTensor, probabilities, [batch_size, seq_len]
        '''
        x_exp = torch.exp(x)
        if mask is not None:
            x_exp = x_exp * mask.float()
        x_sum = torch.sum(x_exp, dim=-1, keepdim=True) + 1e-6
        x_exp /= x_sum
        return x_exp

    def forward(self, x_text, x_attention=None):
        '''
        Forward this module
        :param x_text: torch.FloatTensor, input features, [batch_size, seq_len, D]
        :param mask: torch.ByteTensor, masks for features, [batch_size, seq_len]
        :param x_attention: torch.FloatTensor, input features No. 2 to attent with x_text, [batch_size, seq_len, D]
        :return: torch.FloatTensor, output features, if return sequences, output shape is [batch, SEQ_LEN, D];
                    otherwise output shape is [batch, D]
        '''

        # if x_attention is None:
        #     x_attention = x_text
        # SEQ_LEN = x_text.size()[-2]
        # # view(-1)将张量展平
        # x_attention = x_attention.contiguous().view(-1, self.D)  # [batch_size * seq_len, D]
        # attention = F.tanh(self.linear1(x_attention))  # [batch_size * seq_len, H]
        # attention = self.linear2(attention)  # [batch_size * seq_len, 1]
        # attention = attention.view(-1, SEQ_LEN)  # [batch_size, seq_len]
        # attention = self.softmax_mask(attention, mask)  # [batch_size, seq_len]
        # output = x_text * attention.unsqueeze(-1).expand_as(x_text)  # [batch_size, seq_len, D]
        #
        # if not self.return_sequences:
        #     output = torch.sum(output, -2)
        #     output = output.squeeze(1)
        #
        # return output

        alpha = torch.bmm(x_text, self.linear_W(x_attention).transpose(-1, -2))
        alpha = F.softmax(alpha, dim=-1)
        out = torch.bmm(alpha, x_text)
        return out




if __name__ == "__main__":
    al = AttentionLayer(2, 2, return_sequences=True)
    x = torch.randn(5, 3, 2)
    y = torch.randn(5, 3, 2)

    mask = torch.ByteTensor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0]])

    z = al(x, mask, y)
    print(z)
