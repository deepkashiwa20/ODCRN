import scipy.sparse as ss
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class DataInput(object):
    def __init__(self, data_dir:str):
        self.data_dir = data_dir

    def load_data(self):
        ODPATH = self.data_dir + '/od_day20180101_20210228.npz'
        OD_DAYS = [date.strftime('%Y-%m-%d') for date in pd.date_range(start='2020-01-01', end='2021-02-28', freq='1D')]
        prov_day_data = ss.load_npz(ODPATH)
        prov_day_data_dense = np.array(prov_day_data.todense()).reshape((-1, 47, 47))
        data = prov_day_data_dense[-len(OD_DAYS):,:,:,np.newaxis]
        ODdata = np.log(data + 1.0)      # log transformation
        print(ODdata.shape)
        
        adj = np.load(self.data_dir + '/adjacency_matrix.npy')

        # return a dict
        dataset = dict()
        dataset['OD'] = ODdata
        dataset['adj'] = adj
        return dataset

    
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
        y_seq = torch.from_numpy(np.asarray(y_seq)).float().to(params['GPU'])

        mode_len = self.split2len(data_len=y_seq.shape[0])
        print(mode_len)

        data_loader = dict()        # data_loader for [train, validate, test]
        for mode in ['train', 'validate', 'test']:
            dataset = ODDataset(inputs=feat_dict, output=y_seq, mode=mode, mode_len=mode_len)
            print('Here is data loader', mode, dataset.inputs['x_seq'].shape, dataset.output.shape)
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
    def __init__(self, inputs:dict, output:torch.Tensor, mode:str, mode_len:dict):
        self.mode = mode
        self.mode_len = mode_len
        self.inputs, self.output = self.prepare_xy(inputs, output)

    def __len__(self):
        return self.mode_len[self.mode]

    def __getitem__(self, item):
        return self.inputs['x_seq'][item], self.output[item]

    def prepare_xy(self, inputs:dict, output:torch.Tensor):
        if self.mode == 'train':
            start_idx = 0
        elif self.mode == 'validate':
            start_idx = self.mode_len['train']
        else:       # test
            start_idx = self.mode_len['train']+self.mode_len['validate']

        x = dict()
        x['x_seq'] = inputs['x_seq'][start_idx : (start_idx + self.mode_len[self.mode])]
        y = output[start_idx : start_idx + self.mode_len[self.mode]]
        return x, y


class AdjProcessor():
    def __init__(self, kernel_type:str, K:int):
        self.kernel_type = kernel_type
        # chebyshev (Defferard NIPS'16)/localpool (Kipf ICLR'17)/random_walk_diffusion (Li ICLR'18)
        self.K = K if self.kernel_type != 'localpool' else 1
        # max_chebyshev_polynomial_order (Defferard NIPS'16)/max_diffusion_step (Li ICLR'18)

    def process(self, adj:torch.Tensor):
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
        
        kernels = torch.stack(kernel_list, dim=0)
        return kernels

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

    