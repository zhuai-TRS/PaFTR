import torch
import torch.nn as nn
from torch.nn import functional as F

from layers.SelfAttention_Family import FullAttention, AttentionLayer


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.cycle_len = configs.cycle_len
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = configs.use_revin

        self.valid_fre_points = self.seq_len // 2 + 1
        init_threshold = configs.threshold
        self.freq_threshold = nn.Parameter(torch.tensor(float(init_threshold)))
        self.temperature = 2.0

        self.cycleQuery = torch.nn.Parameter(torch.zeros(self.cycle_len, self.enc_in), requires_grad=True)

        self.channel_attention = AttentionLayer(
            FullAttention(
                mask_flag=False,
                attention_dropout=0.5
            ),
            d_model=self.seq_len,
            n_heads=4
        )

        self.input_proj = nn.Linear(self.seq_len, self.d_model)
        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )
        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len)
        )

        self.use_seq_cycle_complex = configs.use_seq_cycle_complex
        self.fusion_type = configs.fusion_type
        if self.fusion_type == 'time_concat':
            self.emb_layer_time = nn.Linear(self.seq_len * 2, self.seq_len)
        self.qkv = configs.qkv

    def seq_cycle_complex(self, seq, cycle):
        freqs = torch.arange(self.valid_fre_points, device=seq.device).float()
        threshold = F.softplus(self.freq_threshold)
        cycle_weight = torch.sigmoid((threshold - freqs) / self.temperature)
        cycle_weight = cycle_weight.view(1, 1, -1)
        y = cycle_weight * cycle + (1 - cycle_weight) * seq
        return y

    def forward(self, seq_x, cycle_index):
        # instance norm
        if self.use_revin:
            seq_mean = torch.mean(seq_x, dim=1, keepdim=True)
            seq_var = torch.var(seq_x, dim=1, keepdim=True) + 1e-5
            seq_x = (seq_x - seq_mean) / torch.sqrt(seq_var)

        # cycle
        gather_index = (cycle_index.view(-1, 1) + torch.arange(self.seq_len, device=cycle_index.device).view(1,
                                                                                                             -1)) % self.cycle_len
        # B T N
        cycle_x = self.cycleQuery[gather_index]

        # B T N -> B N T
        seq_x = seq_x.permute(0, 2, 1)
        cycle_x = cycle_x.permute(0, 2, 1)

        if self.use_seq_cycle_complex == 'complex':
            if self.fusion_type == 'freq':
                # B N T -> B N fre
                x_fre = torch.fft.rfft(seq_x, dim=-1, norm='ortho')
                x_real, x_imag = x_fre.real, x_fre.imag
                cycle_fre = torch.fft.rfft(cycle_x, dim=-1, norm='ortho')
                cycle_real, cycle_imag = cycle_fre.real, cycle_fre.imag

                y_real = self.seq_cycle_complex(x_real, cycle_real)
                y_imag = self.seq_cycle_complex(x_imag, cycle_imag)
                y = torch.complex(y_real, y_imag)

                # B N fre -> B N T
                x = torch.fft.irfft(y, n=self.seq_len, dim=-1, norm='ortho')
            elif self.fusion_type == 'time_add':
                x = seq_x + cycle_x
            elif self.fusion_type == 'time_concat':
                x = torch.cat([seq_x, cycle_x], dim=-1)
                x = self.emb_layer_time(x)
            else:
                exit('fusion_type: Input error')

        elif self.use_seq_cycle_complex == 'seq':
            x = seq_x
        elif self.use_seq_cycle_complex == 'cycle':
            x = cycle_x
        else:
            exit('seq_cycle_complex: Input error')

        if self.qkv == 'cfs':
            channel_information = self.channel_attention(queries=cycle_x, keys=x, values=seq_x, attn_mask=None)[0]
        elif self.qkv == 'csf':
            channel_information = self.channel_attention(queries=cycle_x, keys=seq_x, values=x, attn_mask=None)[0]
        elif self.qkv == 'fcs':
            channel_information = self.channel_attention(queries=x, keys=cycle_x, values=seq_x, attn_mask=None)[0]
        elif self.qkv == 'fsc':
            channel_information = self.channel_attention(queries=x, keys=seq_x, values=cycle_x, attn_mask=None)[0]
        elif self.qkv == 'sfc':
            channel_information = self.channel_attention(queries=seq_x, keys=x, values=cycle_x, attn_mask=None)[0]
        elif self.qkv == 'scf':
            channel_information = self.channel_attention(queries=seq_x, keys=cycle_x, values=x, attn_mask=None)[0]
        else:
            exit('qkv: Input error')

        input = self.input_proj(seq_x + channel_information)
        hidden = self.model(input)
        output = self.output_proj(hidden + input).permute(0, 2, 1)

        # instance denorm
        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean

        return output