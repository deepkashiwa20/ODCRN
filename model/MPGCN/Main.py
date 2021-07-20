import os
import argparse
import numpy as np
from datetime import datetime
from torch import nn, optim
import torch
import Utils, MPGCN


class ModelTrainer(object):
    def __init__(self, params:dict, data:dict, data_container):
        self.params = params
        self.data_container = data_container
        self.get_static_graph(graph=data['adj'])    # initialize static graphs and K values
        self.model = self.get_model().to(params['GPU'])
        self.criterion = self.get_loss()
        self.optimizer = self.get_optimizer()

    def get_static_graph(self, graph:np.array):
        self.K = self.get_support_K(self.params['kernel_type'], self.params['cheby_order'])
        self.G = self.preprocess_adj(graph, self.params['kernel_type'], self.params['cheby_order'])
        return

    @staticmethod
    def get_support_K(kernel_type, cheby_order):
        if kernel_type == 'localpool':
            assert cheby_order == 1
            K = 1
        elif (kernel_type=='chebyshev')|(kernel_type=='random_walk_diffusion'):
            K = cheby_order + 1
        elif kernel_type == 'dual_random_walk_diffusion':
            K = cheby_order*2 + 1
        else:
            raise ValueError('Invalid kernel_type. Must be one of '
                             '[chebyshev, localpool, random_walk_diffusion, dual_random_walk_diffusion].')
        return K

    def preprocess_adj(self, adj_mtx:np.array, kernel_type, cheby_order):
        self.adj_preprocessor = Utils.AdjProcessor(kernel_type, cheby_order)
        b_adj = torch.from_numpy(adj_mtx).float().unsqueeze(dim=0)      # batch_size=1
        adj = self.adj_preprocessor.process(b_adj)
        return adj.squeeze(dim=0).to(self.params['GPU'])       # G: (support_K, N, N)

    def get_model(self):
        if self.params['model'] == 'MPGCN':
            model = MPGCN.MPGCN(M=2,
                                K=self.K,
                                input_dim=1,
                                lstm_hidden_dim=self.params['hidden_dim'],
                                lstm_num_layers=1,
                                gcn_hidden_dim=self.params['hidden_dim'],
                                gcn_num_layers=3,
                                num_nodes=self.params['N'],
                                out_horizon=self.params['pred_len'],
                                user_bias=True,
                                activation=nn.ReLU)

        else:
            raise NotImplementedError('Invalid model name.')
        return model

    def get_loss(self):
        if self.params['loss'] == 'MSE':
            criterion = nn.MSELoss(reduction='mean')
        elif self.params['loss'] == 'MAE':
            criterion = nn.L1Loss(reduction='mean')
        elif self.params['loss'] == 'Huber':
            criterion = nn.SmoothL1Loss(reduction='mean')
        else:
            raise NotImplementedError('Invalid loss function.')
        return criterion

    def get_optimizer(self):
        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(params=self.model.parameters(),
                                   lr=self.params['learn_rate'])
        else:
            raise NotImplementedError('Invalid optimizer name.')
        return optimizer


    def preprocess_dynamic_graph(self, dyn_G:torch.Tensor):
        # reuse adj_preprocessor initialized in preprocessing static graphs, otherwise needed to initiate one each batch
        return self.adj_preprocessor.process(dyn_G).to(self.params['GPU'])         # (batch, K, N, N)


    def train(self, data_loader:dict, modes:list, early_stop_patience=10):
        checkpoint = {'epoch': 0, 'state_dict': self.model.state_dict()}
        val_loss = np.inf
        patience_count = early_stop_patience

        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print(f'     {self.params["model"]} model training begins:')
        for epoch in range(1, 1 + self.params['num_epochs']):
            running_loss = {mode: 0.0 for mode in modes}
            for mode in modes:
                if mode == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                step = 0
                for x_seq, y_true, O_dyn_G, D_dyn_G in data_loader[mode]:
                    with torch.set_grad_enabled(mode=(mode=='train')):
                        if self.params['model'] == 'MPGCN':
                            dyn_OD_G = (self.preprocess_dynamic_graph(O_dyn_G), self.preprocess_dynamic_graph(D_dyn_G))
                            y_pred = self.model(x_seq=x_seq, G_list=[self.G, dyn_OD_G])
                        else:
                            raise NotImplementedError('Invalid model name.')

                        loss = self.criterion(y_pred, y_true)
                        if mode == 'train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                    running_loss[mode] += loss * y_true.shape[0]    # loss reduction='mean': batchwise average
                    step += y_true.shape[0]
                    torch.cuda.empty_cache()

                # epoch end: evaluate on validation set for early stopping
                if mode == 'validate':
                    epoch_val_loss = running_loss[mode]/step
                    if epoch_val_loss <= val_loss:
                        print(f'Epoch {epoch}, validation loss drops from {val_loss:.5} to {epoch_val_loss:.5}. '
                              f'Update model checkpoint..')
                        val_loss = epoch_val_loss
                        checkpoint.update(epoch=epoch, state_dict=self.model.state_dict())
                        torch.save(checkpoint, self.params['output_dir']+f'/{self.params["model"]}_od.pkl')
                        patience_count = early_stop_patience
                    else:
                        print(f'Epoch {epoch}, validation loss does not improve from {val_loss:.5}.')
                        patience_count -= 1
                        if patience_count == 0:
                            print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
                            print(f'    Early stopping at epoch {epoch}. {self.params["model"]} model training ends.')
                            return

        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print(f'     {self.params["model"]} model training ends.')
        torch.save(checkpoint, self.params['output_dir']+f'/{self.params["model"]}_od.pkl')
        return


    def test(self, data_loader:dict, modes:list):
        trained_checkpoint = torch.load(self.params['output_dir']+f'/{self.params["model"]}_od.pkl')
        self.model.load_state_dict(trained_checkpoint['state_dict'])
        self.model.eval()

        for mode in modes:
            print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
            print(f'     {self.params["model"]} model testing on {mode} data begins:')
            forecast, ground_truth = [], []
            for x_seq, y_true, O_dyn_G, D_dyn_G in data_loader[mode]:
                if self.params['model'] == 'MPGCN':
                    dyn_OD_G = (self.preprocess_dynamic_graph(O_dyn_G), self.preprocess_dynamic_graph(D_dyn_G))
                    y_pred = self.model(x_seq=x_seq, G_list=[self.G, dyn_OD_G])
                else:
                    raise NotImplementedError('Invalid model name.')

                forecast.append(y_pred.cpu().detach().numpy())
                ground_truth.append(y_true.cpu().detach().numpy())

            forecast = np.concatenate(forecast, axis=0)
            ground_truth = np.concatenate(ground_truth, axis=0)
            if mode == 'test':
                np.save(self.params['output_dir'] + '/' + self.params['model'] + '_prediction.npy', forecast)
                np.save(self.params['output_dir'] + '/' + self.params['model'] + '_groundtruth.npy', ground_truth)

            # evaluate on metrics
            MSE, RMSE, MAE, MAPE = self.evaluate(forecast, ground_truth)
            f = open(self.params['output_dir'] + '/' + self.params['model'] + '_prediction_scores.txt', 'a')
            f.write("%s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (mode, MSE, RMSE, MAE, MAPE))
            f.close()
        
        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print(f'     {self.params["model"]} model testing ends.')
        return
    
    @staticmethod
    def evaluate(y_pred: np.array, y_true: np.array, precision=10):
        def MSE(y_pred: np.array, y_true: np.array):
            return np.mean(np.square(y_pred - y_true))
        def RMSE(y_pred:np.array, y_true:np.array):
            return np.sqrt(MSE(y_pred, y_true))
        def MAE(y_pred:np.array, y_true:np.array):
            return np.mean(np.abs(y_pred - y_true))
        def MAPE(y_pred:np.array, y_true:np.array, epsilon=1e-0):       # avoid zero division
            return np.mean(np.abs(y_pred - y_true) / (y_true + epsilon))
        
        print('MSE:', round(MSE(y_pred, y_true), precision))
        print('RMSE:', round(RMSE(y_pred, y_true), precision))
        print('MAE:', round(MAE(y_pred, y_true), precision))
        print('MAPE:', round(MAPE(y_pred, y_true)*100, precision), '%')
        return MSE(y_pred, y_true), RMSE(y_pred, y_true), MAE(y_pred, y_true), MAPE(y_pred, y_true)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run OD Prediction with MPGCN')

    # command line arguments
    parser.add_argument('-GPU', '--GPU', type=str, help='Specify GPU usage', default='cuda:2')
    parser.add_argument('-in', '--input_dir', type=str, default='../../data')
    parser.add_argument('-out', '--output_dir', type=str, default='./output')
    parser.add_argument('-model', '--model', type=str, help='Specify model', choices=['MPGCN'], default='MPGCN')
    parser.add_argument('-obs', '--obs_len', type=int, help='Length of observation sequence', default=7)
    parser.add_argument('-pred', '--pred_len', type=int, help='Length of prediction sequence', default=7)
    parser.add_argument('-split', '--split_ratio', type=int, nargs='+',
                        help='Relative data split ratio in train : validate : test'
                             ' Example: -split 5 1 2', default=[6.4, 1.6, 2])
    parser.add_argument('-batch', '--batch_size', type=int, default=4)
    parser.add_argument('-hidden', '--hidden_dim', type=int, default=32)
    parser.add_argument('-kernel', '--kernel_type', type=str,
                        choices=['chebyshev', 'localpool', 'random_walk_diffusion', 'dual_random_walk_diffusion'],
                        default='random_walk_diffusion')    # GCN kernel type
    parser.add_argument('-K', '--cheby_order', type=int, default=2)    # GCN chebyshev order
    parser.add_argument('-nn', '--nn_layers', type=int, default=2)        # layers
    parser.add_argument('-epoch', '--num_epochs', type=int, default=200)
    parser.add_argument('-loss', '--loss', type=str, help='Specify loss function',
                        choices=['MSE', 'MAE', 'Huber'], default='MSE')
    parser.add_argument('-optim', '--optimizer', type=str, help='Specify optimizer', default='Adam')
    parser.add_argument('-lr', '--learn_rate', type=float, default=1e-4)
    parser.add_argument('-test', '--test_only', type=int, default=0)    # 1 for test only

    params = parser.parse_args().__dict__       # save in dict

    # paths
    os.makedirs(params['output_dir'], exist_ok=True)

    # load data
    data_input = Utils.DataInput(params=params)
    data = data_input.load_data()
    params['N'] = data['OD'].shape[1]

    # get data loader
    data_generator = Utils.DataGenerator(obs_len=params['obs_len'],
                                         pred_len=params['pred_len'], 
                                         data_split_ratio=params['split_ratio'])
    data_loader = data_generator.get_data_loader(data=data, params=params)

    # get model
    trainer = ModelTrainer(params=params, data=data, data_container=data_input)
    
    if bool(params['test_only']) == False:
        trainer.train(data_loader=data_loader,
                      modes=['train', 'validate'])
    trainer.test(data_loader=data_loader,
                 modes=['train', 'test'])


