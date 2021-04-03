import torch
from torch import nn



class BDGCU(nn.Module):        # 2D Graph Convolution Unit
    def __init__(self, K:int, input_dim:int, hidden_dim:int, use_bias=True, activation=None):
        super(BDGCU, self).__init__()
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
        if type(G) == tuple:        # dynamic: O/D graphs different
            assert len(G) == 2
            if (len(G[0].shape) == 2)&(len(G[1].shape) == 2):       # adaptive graph input: (N, N) -> preprocess based on Chebyshev polynomials
                T_k = [[torch.eye(G[i].shape[0]).to(X.device), G[i]] for i in range(2)]       # initialize for O and D
                T_k_pair = []
                for i in range(2):
                    for k in range(2, self.K):
                        T_k[i].append(2 * torch.mm(G[i], T_k[i][-1]) - T_k[i][-2])
                    T_k_pair.append(torch.stack(T_k[i], dim=0))
                G = tuple(T_k_pair)         # adaptive dynamic graph input: ((K, N, N), (K, N, N))

                for o in range(self.K):
                    for d in range(self.K):
                        mode_1_prod = torch.einsum('bncl,nm->bmcl', X, G[0][o, :, :])
                        mode_2_prod = torch.einsum('bmcl,cd->bmdl', mode_1_prod, G[1][d, :, :])
                        feat_set.append(mode_2_prod)
            else:
                raise NotImplementedError
        else:       # static
            assert self.K == G.shape[-3]
            for o in range(self.K):
                for d in range(self.K):
                    mode_1_prod = torch.einsum('bncl,nm->bmcl', X, G[o, :, :])
                    mode_2_prod = torch.einsum('bmcl,cd->bmdl', mode_1_prod, G[d, :, :])
                    feat_set.append(mode_2_prod)

        _2D_feat = torch.cat(feat_set, dim=-1)
        mode_3_prod = torch.einsum('bmdk,kh->bmdh', _2D_feat, self.W)

        if self.use_bias:
            mode_3_prod += self.b
        H = self.activation(mode_3_prod) if self.activation is not None else mode_3_prod
        return H



class BDGCRU_Cell(nn.Module):
    def __init__(self, num_nodes:int, K:int, input_dim:int, hidden_dim:int, use_bias=True, activation=None):
        super(BDGCRU_Cell, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.gates = BDGCU(K, input_dim+hidden_dim, hidden_dim*2, use_bias, activation)
        self.candi = BDGCU(K, input_dim+hidden_dim, hidden_dim, use_bias, activation)

    def init_hidden(self, batch_size:int):
        weight = next(self.parameters()).data
        hidden = (weight.new_zeros(batch_size, self.num_nodes, self.num_nodes, self.hidden_dim))
        return hidden

    def forward(self, G:torch.Tensor, Xt:torch.Tensor, Ht_1:torch.Tensor):
        assert len(Xt.shape) == len(Ht_1.shape) == 4, 'BDGCRU cell must take in 4D tensor as input [Xt, Ht-1]'

        XH = torch.cat([Xt, Ht_1], dim=-1)
        XH_conv = self.gates(X=XH, G=G)

        u, r = torch.split(XH_conv, self.hidden_dim, dim=-1)
        update = torch.sigmoid(u)
        reset = torch.sigmoid(r)

        candi = torch.cat([Xt, reset*Ht_1], dim=-1)
        candi_conv = torch.tanh(self.candi(X=candi, G=G))

        Ht = (1.0 - update) * Ht_1 + update * candi_conv
        return Ht



class BDGCRU_Encoder(nn.Module):
    def __init__(self, num_nodes:int, K:int, input_dim:int, hidden_dim:int, num_layers:int, use_bias=True, activation=None, return_all_layers=True):
        super(BDGCRU_Encoder, self).__init__()
        self.hidden_dim = self._extend_for_multilayers(hidden_dim, num_layers)
        self.num_layers = num_layers
        self.return_all_layers = return_all_layers
        assert len(self.hidden_dim) == self.num_layers, 'Input [hidden, layer] length must be consistent'

        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = input_dim if i==0 else self.hidden_dim[i-1]
            self.cell_list.append(BDGCRU_Cell(num_nodes, K, cur_input_dim, self.hidden_dim[i], use_bias=use_bias, activation=activation))

    def forward(self, G:torch.Tensor or nn.Module, X_seq:torch.Tensor, H0_l=None):
        assert len(X_seq.shape) == 5, 'BDGCRU encoder must take in 5D tensor as input X_seq'
        batch_size, seq_len, _, _, _ = X_seq.shape
        if H0_l is None:
            H0_l = self._init_hidden(batch_size)

        out_seq_lst = list()    # layerwise output seq
        Ht_lst = list()        # layerwise last state
        in_seq_l = X_seq        # current input seq

        for l in range(self.num_layers):
            Ht = H0_l[l]
            out_seq_l = list()
            for t in range(seq_len):
                if (type(G) is not torch.Tensor)&(type(G)!=tuple):     # dyn_adp_G_learner
                    G = G(x_t=X_seq[:, t, :, :, :], device=X_seq.device)      # dynamic: ((N,N), (N,N))
                Ht = self.cell_list[l](G=G, Xt=in_seq_l[:,t,...], Ht_1=Ht)
                out_seq_l.append(Ht)

            out_seq_l = torch.stack(out_seq_l, dim=1)  # (B, T, N, C, h)
            in_seq_l = out_seq_l    # update input seq

            out_seq_lst.append(out_seq_l)
            Ht_lst.append(Ht)

        if not self.return_all_layers:
            out_seq_lst = out_seq_lst[-1:]
            Ht_lst = Ht_lst[-1:]
        return out_seq_lst, Ht_lst

    def _init_hidden(self, batch_size):
        H0_l = []
        for i in range(self.num_layers):
            H0_l.append(self.cell_list[i].init_hidden(batch_size))
        return H0_l

    @staticmethod
    def _extend_for_multilayers(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param



class BDGCRU_Decoder(nn.Module):
    def __init__(self, num_nodes:int, K:int, output_dim:int, hidden_dim:int, num_layers:int, out_horizon:int, use_bias=True, activation=None):
        super(BDGCRU_Decoder, self).__init__()
        self.out_horizon = out_horizon      # output steps
        self.hidden_dim = self._extend_for_multilayers(hidden_dim, num_layers)
        self.num_layers = num_layers
        assert len(self.hidden_dim) == self.num_layers, 'Input [hidden, layer] length must be consistent'

        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = output_dim if i==0 else self.hidden_dim[i-1]
            self.cell_list.append(BDGCRU_Cell(num_nodes, K, cur_input_dim, self.hidden_dim[i], use_bias=use_bias, activation=activation))

    def forward(self, G:torch.Tensor, Xt:torch.Tensor, H0_l:list):
        assert len(Xt.shape) == 4, '2DGCRU decoder must take in 4D tensor as input Xt'

        Ht_lst = list()        # layerwise hidden state
        Xin_l = Xt

        for l in range(self.num_layers):
            Ht_l = self.cell_list[l](G=G, Xt=Xin_l, Ht_1=H0_l[l])
            Ht_lst.append(Ht_l)
            Xin_l = Ht_l      # update input for next layer

        return Ht_l, Ht_lst

    @staticmethod
    def _extend_for_multilayers(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param



class BDGCRU_Decoder_2(nn.Module):          # mixed encoder: layer-1 adj, layer-2 dyn_adp
    def __init__(self, num_nodes:int, K:int, output_dim:int, hidden_dim:int, out_horizon:int, num_layers:int=2, use_bias=True, activation=None):
        super(BDGCRU_Decoder_2, self).__init__()
        self.out_horizon = out_horizon      # output steps
        self.num_layers = num_layers

        self.cell_list = nn.ModuleList()
        self.cell_list.append(BDGCRU_Cell(num_nodes, K, output_dim, hidden_dim, use_bias=use_bias, activation=activation))      # layer-1
        self.cell_list.append(BDGCRU_Cell(num_nodes, K, hidden_dim, hidden_dim, use_bias=use_bias, activation=activation))      # layer-2

    def forward(self, G_list:list, Xt:torch.Tensor, H0_l:list):
        assert len(Xt.shape) == 4, '2DGCRU decoder must take in 4D tensor as input Xt'
        assert self.num_layers == len(G_list)

        Ht_lst = list()        # layerwise hidden state
        Xin_l = Xt
        for l in range(self.num_layers):
            if type(G_list[l]) is not torch.Tensor:     # dyn_adp_G_learner
                G_list[l] = G_list[l](x_t=Xt[:, :, :, :], device=Xt.device)  # dynamic: ((N,N), (N,N))
            Ht_l = self.cell_list[l](G=G_list[l], Xt=Xin_l, Ht_1=H0_l[l])
            Ht_lst.append(Ht_l)
            Xin_l = Ht_l      # update input for next layer

        return Ht_l, Ht_lst



class BDGCRN_Jiang(nn.Module):
    def __init__(self, num_nodes:int, K:int, input_dim:int, hidden_dim:int, out_horizon:int, num_layers:int, use_bias=True, activation=None):
        super(BDGCRN_Jiang, self).__init__()
        self.encoder_1 = BDGCRU_Encoder(num_nodes, K, input_dim, hidden_dim, 1, use_bias, activation, return_all_layers=False)
        self.bn_1 = nn.LayerNorm(normalized_shape=[7,num_nodes,num_nodes,hidden_dim])
        self.encoder_2 = BDGCRU_Encoder(num_nodes, K, hidden_dim, hidden_dim, 1, use_bias, activation, return_all_layers=False)
        self.bn_2 = nn.LayerNorm(normalized_shape=[7,num_nodes,num_nodes,hidden_dim])

        self.dyn_adp_G_learner = DynGraph_Learner(num_nodes=num_nodes, emb_dim=10)

        self.decoder_1 = BDGCRU_Encoder(num_nodes, K, hidden_dim, hidden_dim, 1, use_bias, activation, return_all_layers=False)
        self.bn_3 = nn.LayerNorm(normalized_shape=[7,num_nodes,num_nodes,hidden_dim])
        self.decoder_2 = BDGCRU_Encoder(num_nodes, K, hidden_dim, input_dim, 1, use_bias, activation, return_all_layers=False)
        self.out_horizon = out_horizon

    def forward(self, G:torch.Tensor, X_seq:torch.Tensor):
        assert len(X_seq.shape) == 5, 'BDGCRN must take in 5D tensor as input X_seq: [4,7,47,47,1]'

        # encoding
        h_seq_lst, H_lst_1 = self.encoder_1(G=self.dyn_adp_G_learner, X_seq=X_seq, H0_l=None)
        h_seq_lst = self.bn_1(h_seq_lst[-1])
        h_seq_lst, H_lst_2 = self.encoder_2(G=G, X_seq=h_seq_lst, H0_l=None)
        h_seq_lst = self.bn_2(h_seq_lst[-1])

        # decoding
        deco_input = torch.stack([h_seq_lst[:,-1,:,:,:]]*self.out_horizon, dim=1)
        #Ht_lst = H_lst_1 + H_lst_2
        h_seq_lst, H_lst_1 = self.decoder_1(G=self.dyn_adp_G_learner, X_seq=deco_input, H0_l=None)
        h_seq_lst = self.bn_3(h_seq_lst[-1])
        out_seq_lst, H_lst_2 = self.decoder_2(G=G, X_seq=h_seq_lst, H0_l=None)

        return torch.relu(out_seq_lst[-1])



class BDGCRN(nn.Module):
    def __init__(self, num_nodes:int, K:int, input_dim:int, hidden_dim:int, out_horizon:int, num_layers:int, use_bias=True, activation=None):
        super(BDGCRN, self).__init__()
        self.encoder_1 = BDGCRU_Encoder(num_nodes, K, input_dim, hidden_dim, 1, use_bias, activation, return_all_layers=False)
        self.encoder_2 = BDGCRU_Encoder(num_nodes, K, hidden_dim, hidden_dim, 1, use_bias, activation, return_all_layers=False)
        self.dyn_adp_G_learner = DynGraph_Learner(num_nodes=num_nodes, emb_dim=10)

        #self.decoder = BDGCRU_Decoder(num_nodes, K, input_dim, hidden_dim, num_layers, out_horizon, use_bias, activation)
        self.decoder = BDGCRU_Decoder_2(num_nodes, K, input_dim, hidden_dim, out_horizon, num_layers, use_bias, activation)
        #self.out_projector = nn.Sequential(nn.Linear(in_features=hidden_dim, out_features=input_dim, bias=use_bias), nn.ReLU())
        self.out_projector = nn.Linear(in_features=hidden_dim, out_features=input_dim, bias=use_bias)

    def forward(self, G:torch.Tensor, X_seq:torch.Tensor):
        assert len(X_seq.shape) == 5, 'BDGCRN must take in 5D tensor as input X_seq: [4,7,47,47,1]'

        # encoding
        out_seq_lst, H_lst_1 = self.encoder_1(G=self.dyn_adp_G_learner, X_seq=X_seq, H0_l=None)
        out_seq_lst, H_lst_2 = self.encoder_2(G=G, X_seq=out_seq_lst[-1], H0_l=None)

        # initiate decoder input
        #deco_input = self.out_projector(H_lst_2[-1])
        deco_input = torch.zeros((X_seq.shape[0], X_seq.shape[2], X_seq.shape[3], X_seq.shape[4]), device=X_seq.device)
        Ht_lst = H_lst_1 + H_lst_2

        # decoding
        outputs = list()
        for t in range(self.decoder.out_horizon):
            Ht_l, Ht_lst = self.decoder(G_list=[self.dyn_adp_G_learner, G], Xt=deco_input, H0_l=Ht_lst)
            output = self.out_projector(Ht_l)
            deco_input = output     # update decoder input
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)  # (B, horizon, N, C, h)
        return outputs



class DynGraph_Learner(nn.Module):          # parameterize a dynamic graph
    def __init__(self, num_nodes:int, emb_dim:int):
        super(DynGraph_Learner, self).__init__()
        self.num_nodes = num_nodes
        self.emb_dim = emb_dim

        self.init_2_params_SLCNN()
        #self.init_params_DMGCRN()

    # todo: try initialize with adj??
    def init_2_params_SLCNN(self):
        self.W_o = nn.Parameter(torch.empty(self.num_nodes, self.num_nodes), requires_grad=True)    # weight of graph learning for origins
        nn.init.xavier_normal_(self.W_o)
        self.W_d = nn.Parameter(torch.empty(self.num_nodes, self.num_nodes), requires_grad=True)    # weight of graph learning for destinations
        nn.init.xavier_normal_(self.W_d)
        return

    def forward(self, x_t:torch.Tensor, device:str):
        assert len(x_t.shape)==4    # (batch, N, N, hidden)
        #x_t = x_t.squeeze(dim=-1)
        O = torch.softmax(torch.relu(torch.einsum('bpdh,dd,bqdh->pq', x_t, self.W_o.to(device), x_t)), dim=1)
        D = torch.softmax(torch.relu(torch.einsum('boeh,oo,bofh->ef', x_t, self.W_d.to(device), x_t)), dim=1)
        return (O, D)

    def forward_sparse(self, x_t:torch.Tensor, device:str, alpha:int=3, k:int=20):
        assert len(x_t.shape)==4    # (batch, N, N, 1)
        #x_t = x_t.squeeze(dim=-1)
        O = torch.softmax(torch.relu(torch.tanh(alpha * torch.einsum('bpdh,dd,bqdh->pq', x_t, self.W_o.to(device), x_t))), dim=1)
        D = torch.softmax(torch.relu(torch.tanh(alpha * torch.einsum('boeh,oo,bofh->ef', x_t, self.W_d.to(device), x_t))), dim=1)

        mask_O = torch.zeros(x_t.shape[1], x_t.shape[1]).to(device)
        mask_O.fill_(float('0'))
        s1,t1 = O.topk(k,1)
        mask_O.scatter_(1,t1,s1.fill_(1))
        O_G = O*mask_O

        mask_D = torch.zeros(x_t.shape[2], x_t.shape[2]).to(device)
        mask_D.fill_(float('0'))
        s2,t2 = D.topk(k,1)
        mask_D.scatter_(1,t2,s2.fill_(1))
        D_G = D*mask_D

        return (O_G, D_G)

    def init_params_DMGCRN(self):
        self.W_o_l = nn.Parameter(torch.empty(self.num_nodes, self.emb_dim), requires_grad=True)
        nn.init.xavier_normal_(self.W_o_l)
        #self.W_o_r = nn.Parameter(torch.empty(self.num_nodes, self.emb_dim), requires_grad=True)
        #nn.init.xavier_normal_(self.W_o_r)

        self.W_d_l = nn.Parameter(torch.empty(self.num_nodes, self.emb_dim), requires_grad=True)
        nn.init.xavier_normal_(self.W_d_l)
        #self.W_d_r = nn.Parameter(torch.empty(self.num_nodes, self.emb_dim), requires_grad=True)
        #nn.init.xavier_normal_(self.W_d_r)
        return

    def forward_DMGCRN(self, x_t:torch.Tensor, device:str, alpha:int=3, k:int=20):
        assert len(x_t.shape) == 4  # (batch, N, N, 1)
        x_t = x_t.squeeze(dim=-1)
        #O_G = torch.softmax(torch.relu(torch.einsum('bpd,dh,hd,bqd->pq', x_t, self.W_o_l.to(device), self.W_o_r.t().to(device), x_t)), dim=1)
        #D_G = torch.softmax(torch.relu(torch.einsum('boe,oh,ho,bof->ef', x_t, self.W_d_l.to(device), self.W_d_r.t().to(device), x_t)), dim=1)

        O_l = torch.tanh(alpha * torch.einsum('bpd,dh->bph', x_t, self.W_o_l.to(device)))
        O_r = torch.tanh(alpha * torch.einsum('hd,bqd->bqh', self.W_o_l.t().to(device), x_t))
        O = torch.einsum('bph,bqh->pq', O_l, O_r) - torch.einsum('bqh,bph->qp', O_r, O_l)
        O_G = torch.relu(torch.tanh(alpha * O))

        mask_O = torch.zeros(x_t.shape[1], x_t.shape[1]).to(device)
        mask_O.fill_(float('0'))
        s1,t1 = O_G.topk(k,1)
        mask_O.scatter_(1,t1,s1.fill_(1))
        O_G = O_G*mask_O

        D_l = torch.tanh(alpha * torch.einsum('bpd,dh->bph', x_t, self.W_d_l.to(device)))
        D_r = torch.tanh(alpha * torch.einsum('hd,bqd->bqh', self.W_d_l.t().to(device), x_t))
        D = torch.einsum('bph,bqh->pq', D_l, D_r) - torch.einsum('bqh,bph->qp', D_r, D_l)
        D_G = torch.relu(torch.tanh(alpha * D))

        mask_D = torch.zeros(x_t.shape[2], x_t.shape[2]).to(device)
        mask_D.fill_(float('0'))
        s2,t2 = D_G.topk(k,1)
        mask_D.scatter_(1,t2,s2.fill_(1))
        D_G = D_G*mask_D

        return (O_G, D_G)

