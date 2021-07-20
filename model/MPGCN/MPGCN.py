import torch
from torch import nn



class BDGCN(nn.Module):        # 2DGCN: handling both static and dynamic graph input
    def __init__(self, K:int, input_dim:int, hidden_dim:int, use_bias=True, activation=None):
        super(BDGCN, self).__init__()
        self.K = K
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.activation = activation() if activation is not None else None
        self.init_params()

    def init_params(self, b_init=0.0):
        self.W = nn.Parameter(torch.empty(self.input_dim*(self.K**2), self.hidden_dim), requires_grad=True)
        nn.init.xavier_normal_(self.W)
        if self.use_bias:
            self.b = nn.Parameter(torch.empty(self.hidden_dim), requires_grad=True)
            nn.init.constant_(self.b, val=b_init)
        return

    def forward(self, X:torch.Tensor, G:torch.Tensor or tuple):
        feat_set = list()
        if type(G) == torch.Tensor:         # static graph input: (K, N, N)
            assert self.K == G.shape[-3]
            for o in range(self.K):
                for d in range(self.K):
                    mode_1_prod = torch.einsum('bncl,nm->bmcl', X, G[o, :, :])
                    mode_2_prod = torch.einsum('bmcl,cd->bmdl', mode_1_prod, G[d, :, :])
                    feat_set.append(mode_2_prod)

        elif type(G) == tuple:              # dynamic graph input: ((batch, K, N, N), (batch, K, N, N))
            assert (len(G) == 2) & (self.K == G[0].shape[-3] == G[1].shape[-3])
            for o in range(self.K):
                for d in range(self.K):
                    mode_1_prod = torch.einsum('bncl,bnm->bmcl', X, G[0][:, o, :, :])
                    mode_2_prod = torch.einsum('bmcl,bcd->bmdl', mode_1_prod, G[1][:, d, :, :])
                    feat_set.append(mode_2_prod)
        else:
            raise NotImplementedError

        _2D_feat = torch.cat(feat_set, dim=-1)
        mode_3_prod = torch.einsum('bmdk,kh->bmdh', _2D_feat, self.W)

        if self.use_bias:
            mode_3_prod += self.b
        H = self.activation(mode_3_prod) if self.activation is not None else mode_3_prod
        return H



class MPGCN(nn.Module):
    def __init__(self, M:int, K:int, input_dim:int, lstm_hidden_dim:int, lstm_num_layers:int, gcn_hidden_dim:int, gcn_num_layers:int,
                 num_nodes:int, out_horizon:int, user_bias:bool, activation=None):
        super(MPGCN, self).__init__()
        self.M = M      # input graphs
        self.K = K      # chebyshev order
        self.num_nodes = num_nodes
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.gcn_num_layers = gcn_num_layers
        self.out_horizon = out_horizon

        # initiate a branch of (LSTM, 2DGCN, FC) for each graph input
        self.branch_models = nn.ModuleList()
        for m in range(self.M):
            branch = nn.ModuleDict()
            branch['temporal'] = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_dim, num_layers=lstm_num_layers, batch_first=True)
            branch['spatial'] = nn.ModuleList()
            for n in range(gcn_num_layers):
                cur_input_dim = lstm_hidden_dim if n == 0 else gcn_hidden_dim
                branch['spatial'].append(BDGCN(K=K, input_dim=cur_input_dim, hidden_dim=gcn_hidden_dim, use_bias=user_bias, activation=activation))
            branch['fc'] = nn.Sequential(
                nn.Linear(in_features=gcn_hidden_dim, out_features=input_dim, bias=True),
                nn.ReLU())
            branch['decoder'] = stepwise_lstm_decoder(out_horizon=out_horizon, hidden_dim=lstm_hidden_dim,
                                                      num_layers=lstm_num_layers, input_dim=input_dim,
                                                      use_bias=user_bias)
            self.branch_models.append(branch)


    def init_hidden_list(self, batch_size:int):     # for LSTM initialization
        hidden_list = list()
        for m in range(self.M):
            weight = next(self.parameters()).data
            hidden = (weight.new_zeros(self.lstm_num_layers, batch_size * (self.num_nodes**2), self.lstm_hidden_dim),
                      weight.new_zeros(self.lstm_num_layers, batch_size * (self.num_nodes**2), self.lstm_hidden_dim))
            hidden_list.append(hidden)
        return hidden_list

    def forward(self, x_seq: torch.Tensor, G_list:list):
        '''
        :param x_seq: (batch, seq, O, D, 1)
        :param G_list: static graph (K, N, N); dynamic OD graph tuple ((batch, K, N, N), (batch, K, N, N))
        :return:
        '''
        assert (len(x_seq.shape) == 5)&(self.num_nodes == x_seq.shape[2] == x_seq.shape[3])
        assert len(G_list) == self.M
        batch_size, seq_len, _, _, i = x_seq.shape
        hidden_list = self.init_hidden_list(batch_size)

        lstm_in = x_seq.permute(0, 2, 3, 1, 4).reshape(batch_size*(self.num_nodes**2), seq_len, i)
        branch_out = list()
        for m in range(self.M):
            lstm_out, hidden_list[m] = self.branch_models[m]['temporal'](lstm_in, hidden_list[m])
            gcn_in = lstm_out[:,-1,:].reshape(batch_size, self.num_nodes, self.num_nodes, self.lstm_hidden_dim)
            for n in range(self.gcn_num_layers):
                gcn_in = self.branch_models[m]['spatial'][n](gcn_in, G_list[m])
            fc_out = self.branch_models[m]['fc'](gcn_in)
            #branch_out.append(fc_out)

            # stepwise decoding
            decoder_in = fc_out.reshape(batch_size*(self.num_nodes**2), i)    # initialize input for lstm decoder
            hidden = [(hidden_list[m][0][l,:,:], hidden_list[m][1][l,:,:]) for l in range(self.lstm_num_layers)]    # initialize hidden
            decoder_out = []
            for t in range(self.out_horizon):
                ht, hidden = self.branch_models[m]['decoder'](Xt=decoder_in, H0_l=hidden)     # copy encoder's hidden states
                step_out = self.branch_models[m]['fc'](ht.reshape(batch_size, self.num_nodes, self.num_nodes, self.lstm_hidden_dim))
                decoder_in = step_out.reshape(batch_size*(self.num_nodes**2), i)
                decoder_out.append(step_out)
            decoder_out = torch.stack(decoder_out, dim=1)       # (batch, horizon, N, N, 1)
            branch_out.append(decoder_out)

        # ensemble
        ensemble_out = torch.mean(torch.stack(branch_out, dim=-1), dim=-1)
        return ensemble_out



class stepwise_lstm_decoder(nn.Module):      # for multistep output of each MPGCN branch
    def __init__(self, out_horizon:int, hidden_dim:int, num_layers:int, input_dim:int, use_bias=True):
        super(stepwise_lstm_decoder, self).__init__()
        self.out_horizon = out_horizon  # output steps
        self.hidden_dim = self._extend_for_multilayers(hidden_dim, num_layers)
        self.num_layers = num_layers
        assert len(self.hidden_dim) == self.num_layers, 'Input [hidden, layer] length must be consistent'

        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = input_dim if i == 0 else self.hidden_dim[i - 1]
            self.cell_list.append(nn.LSTMCell(input_size=cur_input_dim, hidden_size=hidden_dim, bias=use_bias))

    def forward(self, Xt: torch.Tensor, H0_l: list):
        assert len(Xt.shape) == 2, 'LSTM decoder must take in 2D tensor as input Xt'

        Ht_lst = list()     # layerwise hidden states: (h, c)
        Xin_l = Xt
        for l in range(self.num_layers):
            Ht_l, Ct_l = self.cell_list[l](Xin_l, H0_l[l])
            Ht_lst.append((Ht_l, Ct_l))
            Xin_l = Ht_l    # update input for next layer

        return Ht_l, Ht_lst

    @staticmethod
    def _extend_for_multilayers(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param