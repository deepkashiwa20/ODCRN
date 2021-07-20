import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import distance
import scipy.sparse as ss



class DataInput(object):
    def __init__(self, params:dict):
        self.params = params

    def load_data(self):
        ODPATH = self.params['input_dir'] + '/od_day20180101_20210228.npz'
        OD_DAYS = [date.strftime('%Y-%m-%d') for date in pd.date_range(start='2020-01-01', end='2021-02-28', freq='1D')]
        prov_day_data = ss.load_npz(ODPATH)
        prov_day_data_dense = np.array(prov_day_data.todense()).reshape((-1, 47, 47))
        data = prov_day_data_dense[-len(OD_DAYS):, :, :, np.newaxis]
        ODdata = np.log(data + 1.0)        # log transformation
        print(ODdata.shape)
        
        adj = np.load(self.params['input_dir'] + '/adjacency_matrix.npy')
        
        # return a dict
        dataset = dict()
        dataset['OD'] = ODdata
        dataset['adj'] = adj
        dataset['O_dyn_G'], dataset['D_dyn_G'] = self.construct_dyn_G(data)    # use unnormalized OD

        return dataset

    def construct_dyn_G(self, OD_data:np.array, perceived_period:int=7):        # construct dynamic graphs based on OD history
        train_len = int(OD_data.shape[0] * self.params['split_ratio'][0] / sum(self.params['split_ratio']))
        num_periods_in_history = train_len // perceived_period      # dump the remainder
        OD_history = OD_data[:num_periods_in_history * perceived_period, :,:,:]

        O_dyn_G, D_dyn_G = [], []
        for t in range(perceived_period):
            OD_t_avg = np.mean(OD_history[t::perceived_period,:,:,:], axis=0).squeeze(axis=-1)
            O, D = OD_t_avg.shape

            O_G_t = np.zeros((O, O))    # initialize O graph at t
            for i in range(O):
                for j in range(O):
                    O_G_t[i, j] = distance.cosine(OD_t_avg[i,:], OD_t_avg[j,:])     # eq (6)
            D_G_t = np.zeros((D, D))    # initialize D graph at t
            for i in range(D):
                for j in range(D):
                    D_G_t[i, j] = distance.cosine(OD_t_avg[:,i], OD_t_avg[j,:])     # eq (7)
            O_dyn_G.append(O_G_t), D_dyn_G.append(D_G_t)

        return np.stack(O_dyn_G, axis=-1), np.stack(D_dyn_G, axis=-1)


class DataGenerator(object):
    def __init__(self, obs_len:int, pred_len, data_split_ratio:tuple):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.data_split_ratio = data_split_ratio

    def split2len(self, data_len:int):
        mode_len = dict()
        mode_len['train'] = int(self.data_split_ratio[0]/sum(self.data_split_ratio) * data_len)
        mode_len['validate'] = int(self.data_split_ratio[1]/sum(self.data_split_ratio) * data_len)
        mode_len['test'] = data_len - mode_len['train'] - mode_len['validate']
        return mode_len

    def get_data_loader(self, data:dict, params:dict):
        x_seq, y_seq = self.get_feats(data['OD'])

        feat_dict = dict()
        feat_dict['x_seq'] = torch.from_numpy(np.asarray(x_seq)).float().to(params['GPU'])
        feat_dict['O_dyn_G'], feat_dict['D_dyn_G'] = torch.from_numpy(data['O_dyn_G']).float(), torch.from_numpy(data['D_dyn_G']).float()
        y_seq = torch.from_numpy(np.asarray(y_seq)).float().to(params['GPU'])

        mode_len = self.split2len(data_len=y_seq.shape[0])

        data_loader = dict()        # data_loader for [train, validate, test]
        for mode in ['train', 'validate', 'test']:
            dataset = ODDataset(inputs=feat_dict, output=y_seq,
                                mode=mode, mode_len=mode_len, obs_len=self.obs_len)
            data_loader[mode] = DataLoader(dataset=dataset, batch_size=params['batch_size'], shuffle=True)
        # data loading default: single-processing | for multi-processing: num_workers=pos_int or pin_memory=True (GPU)
        # data_loader multi-processing
        return data_loader

    def get_feats(self, data:np.array):
        x, y = [], []
        for i in range(self.obs_len, data.shape[0]-self.pred_len):
            x.append(data[i-self.obs_len : i])
            y.append(data[i : i+self.pred_len])
        return x, y


class ODDataset(Dataset):
    def __init__(self, inputs:dict, output:torch.Tensor, mode:str, mode_len:dict, obs_len:int):
        self.mode = mode
        self.mode_len = mode_len
        self.inputs, self.output = self.prepare_xy(inputs, output)
        self.obs_len = obs_len

    def __len__(self):
        return self.mode_len[self.mode]

    def __getitem__(self, item:int):        # item: time index in current mode
        O_G_t, D_G_t = self.timestamp_query(self.inputs['O_dyn_G'], self.inputs['D_dyn_G'], item)
        return self.inputs['x_seq'][item], self.output[item], O_G_t, D_G_t      # dynamic graph shape: (batch, N, N)

    def timestamp_query(self, O_dyn_G:torch.Tensor, D_dyn_G:torch.Tensor, t:int, perceived_period:int=7):    # for dynamic graph at t
        # get y's timestamp relative to initial timestamp of the dataset
        if self.mode == 'train':
            timestamp = self.obs_len + t
        elif self.mode == 'validate':
            timestamp = self.obs_len + self.mode_len['train'] + t
        else:       # test
            timestamp = self.obs_len + self.mode_len['train'] + self.mode_len['validate'] + t

        key = timestamp % perceived_period
        O_G_t, D_G_t = O_dyn_G[:,:,key], D_dyn_G[:,:,key]
        return O_G_t, D_G_t

    def prepare_xy(self, inputs:dict, output:torch.Tensor):
        if self.mode == 'train':
            start_idx = 0
        elif self.mode == 'validate':
            start_idx = self.mode_len['train']
        else:       # test
            start_idx = self.mode_len['train']+self.mode_len['validate']

        x = dict()
        x['x_seq'] = inputs['x_seq'][start_idx : (start_idx + self.mode_len[self.mode])]
        x['O_dyn_G'], x['D_dyn_G'] = inputs['O_dyn_G'], inputs['D_dyn_G']
        y = output[start_idx : start_idx + self.mode_len[self.mode]]
        return x, y


class AdjProcessor():
    def __init__(self, kernel_type:str, K:int):
        self.kernel_type = kernel_type
        # chebyshev (Defferard NIPS'16)/localpool (Kipf ICLR'17)/random_walk_diffusion (Li ICLR'18)
        self.K = K if self.kernel_type != 'localpool' else 1
        # max_chebyshev_polynomial_order (Defferard NIPS'16)/max_diffusion_step (Li ICLR'18)

    def process(self, flow:torch.Tensor):
        '''
        Generate adjacency matrices
        :param flow: batch flow stat - (batch_size, Origin, Destination) torch.Tensor
        :return: processed adj matrices - (batch_size, K_supports, O, D) torch.Tensor
        '''
        batch_list = list()

        for b in range(flow.shape[0]):
            adj = flow[b, :, :]
            kernel_list = list()

            if self.kernel_type in ['localpool', 'chebyshev']:  # spectral
                adj_norm = self.symmetric_normalize(adj)
                if self.kernel_type == 'localpool':
                    localpool = torch.eye(adj_norm.shape[0]) + adj_norm  # same as add self-loop first
                    kernel_list.append(localpool)

                else:  # chebyshev
                    laplacian_norm = torch.eye(adj_norm.shape[0]) - adj_norm
                    laplacian_rescaled = self.rescale_laplacian(laplacian_norm)
                    kernel_list = self.compute_chebyshev_polynomials(laplacian_rescaled, kernel_list)

            elif self.kernel_type == 'random_walk_diffusion':  # spatial
                # diffuse k steps on transition matrix P
                P_forward = self.random_walk_normalize(adj)
                kernel_list = self.compute_chebyshev_polynomials(P_forward.T, kernel_list)

            elif self.kernel_type == 'dual_random_walk_diffusion':
                # diffuse k steps bidirectionally on transition matrix P
                P_forward = self.random_walk_normalize(adj)
                P_backward = self.random_walk_normalize(adj.T)
                forward_series, backward_series = [], []
                forward_series = self.compute_chebyshev_polynomials(P_forward.T, forward_series)
                backward_series = self.compute_chebyshev_polynomials(P_backward.T, backward_series)
                kernel_list += forward_series + backward_series[1:]  # 0-order Chebyshev polynomial is same: I

            else:
                raise ValueError('Invalid kernel_type. Must be one of [chebyshev, localpool, random_walk_diffusion, dual_random_walk_diffusion].')

            # print(f"Minibatch {b}: {self.kernel_type} kernel has {len(kernel_list)} support kernels.")
            kernels = torch.stack(kernel_list, dim=0)
            batch_list.append(kernels)
        batch_adj = torch.stack(batch_list, dim=0)
        return batch_adj

    @staticmethod
    def random_walk_normalize(A):   # asymmetric
        d_inv = torch.pow(A.sum(dim=1), -1)   # OD matrix Ai,j sum on j (axis=1)
        d_inv[torch.isinf(d_inv)] = 0.
        D = torch.diag(d_inv)
        P = torch.mm(D, A)
        return P

    @staticmethod
    def symmetric_normalize(A):
        D = torch.diag(torch.pow(A.sum(dim=1), -0.5))
        A_norm = torch.mm(torch.mm(D, A), D)
        return A_norm

    @staticmethod
    def rescale_laplacian(L):
        # rescale laplacian to arccos range [-1,1] for input to Chebyshev polynomials of the first kind
        try:
            lambda_ = torch.eig(L)[0][:,0]      # get the real parts of eigenvalues
            lambda_max = lambda_.max()      # get the largest eigenvalue
        except:
            print("Eigen_value calculation didn't converge, using max_eigen_val=2 instead.")
            lambda_max = 2
        L_rescaled = (2 / lambda_max) * L - torch.eye(L.shape[0])
        return L_rescaled

    def compute_chebyshev_polynomials(self, x, T_k):
        # compute Chebyshev polynomials up to order k. Return a list of matrices.
        # print(f"Computing Chebyshev polynomials up to order {self.K}.")
        for k in range(self.K + 1):
            if k == 0:
                T_k.append(torch.eye(x.shape[0]))
            elif k == 1:
                T_k.append(x)
            else:
                T_k.append(2 * torch.mm(x, T_k[k-1]) - T_k[k-2])
        return T_k

