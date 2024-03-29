import sys
sys.path.insert(0,'./keops')

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["USE_KEOPS"] = "False"

import pickle
import numpy as np

import random

import torch

from torch.utils.data import DataLoader
from datasets import PlanetoidDataset, TadpoleDataset
import pytorch_lightning as pl
from DGMlib.model_dDGM import DGM_Model

from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

with open('data/Datos/x_numerical.pkl', 'rb') as f:
    x_numeric = pickle.load(f)

with open('data/Datos/x_textual.pkl', 'rb') as f:
    x_text = pickle.load(f)

with open('data/Datos/y_.pkl', 'rb') as f:
    y = pickle.load(f)

y = np.expand_dims(y,axis = 2)

x = np.concatenate((x_numeric, x_text), axis = 2)
datos = np.concatenate((x,y), axis = 2)

test_size = int(0.2 * 730)

time_indices = list(range(730))
random.shuffle(time_indices)

test_indices = time_indices[:test_size]
train_indices = time_indices[test_size:]

xTr = x[train_indices]
xTe = x[test_indices]

data=x.reshape(-1, 13)

def run_training_process(run_params):
    
    train_data = None
    test_data = None
    
    if run_params.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        train_data = PlanetoidDataset(split='train', name=run_params.dataset, device='cuda')
        val_data = PlanetoidDataset(split='val', name=run_params.dataset, samples_per_epoch=1)
        test_data = PlanetoidDataset(split='test', name=run_params.dataset, samples_per_epoch=1)
        
    if run_params.dataset == 'tadpole':
        train_data = TadpoleDataset(fold=run_params.fold,train=True, device='cuda')
        val_data = test_data = TadpoleDataset(fold=run_params.fold, train=False,samples_per_epoch=1)
    
    if run_params.dataset == "datos":
        #Esto crea un objeto como lo necesitamos
        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, data, targets, split='train', name=None, device='cuda'):
                self.data = torch.tensor(data, dtype=torch.float32)
                self.targets = torch.tensor(targets, dtype=torch.float32)
                self.split = split
                self.name = name
                self.device = 'cuda'
                self.n_samples = data.shape[0] 
                self.n_features = 13
                self.num_classes=1
                self.edge_index = torch.tensor([(i,j) for i in range(198) for j in range(198) if i!=j], dtype=torch.long).t().cpu()
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, index):
                sample = self.data[index]
                target = self.targets[index]
                
                mask = torch.ones_like(sample)
                mask[sample == 0] = 0

                # Compute edges tensor
                edges = self.edge_index
                
                return sample, target, mask, edges
                    
        train_data = CustomDataset(data=xTr, targets=y, split='train', name='my_dataset', device='cuda')
        val_data = CustomDataset(data=xTe, targets=y, split='test', name='my_dataset')
        test_data = CustomDataset(data=xTe, targets=y, split='validation', name='my_dataset')

        #class CustomDataset(torch.utils.data.Dataset):
        #    def __init__(self, data, targets):
         #       self.data = data
          #      self.targets = targets
           #     
           # def __len__(self):
            #    return len(self.targets)
                
           # def __getitem__(self, index):
            #    x = torch.FloatTensor(self.data[index])
             #   y = torch.LongTensor([self.targets[index]])
              #  return x, y
        
        
        #### Hacer esto con Train, Test y Validation, algo de tipo:
            #train_data=CustomDataset(xTrain, yTrain)
            #val_data=CustomDataset(xVal, yVal)
            #test_data=CustomDataset(xVal, yVal)
        #custom_dataset = CustomDataset(x, y)
        
    if train_data is None:
        raise Exception("Dataset %s not supported" % run_params.dataset)
        
    train_loader = DataLoader(train_data, batch_size=8,num_workers=0)
    val_loader = DataLoader(val_data, batch_size=8)
    test_loader = DataLoader(test_data, batch_size=8)

    class MyDataModule(pl.LightningModule): 
        def setup(self,stage=None):
            pass
        def train_dataloader(self):
            return train_loader
        def val_dataloader(self):
            return val_loader
        def test_dataloader(self):
            return test_loader
    
    #configure input feature size
    if run_params.pre_fc is None or len(run_params.pre_fc)==0: 
        if len(run_params.dgm_layers[0])>0:
            run_params.dgm_layers[0][0]=train_data.n_features
        run_params.conv_layers[0][0]=train_data.n_features
    else:
        run_params.pre_fc[0]=train_data.n_features
    run_params.fc_layers[-1] = train_data.num_classes
    
    model = DGM_Model(run_params)

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=1,
        patience=20,
        verbose=False,
        mode='min')
    callbacks = [checkpoint_callback,early_stop_callback]
    
    if val_data==test_data:
        callbacks = None
        
    logger = TensorBoardLogger("logs/")
    trainer = pl.Trainer.from_argparse_args(run_params,logger=logger,
                                            callbacks=callbacks)
    
    trainer.fit(model, datamodule=MyDataModule())
    trainer.test(dataloaders=(test_loader))
    
if __name__ == "__main__":

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    params = parser.parse_args(['--gpus','1',                         
                              '--log_every_n_steps','2',                          
                              '--max_epochs','20',
                              '--check_val_every_n_epoch','2'])
    parser.add_argument("--num_gpus", default=10, type=int)
    
    parser.add_argument("--dataset", default='datos')
    parser.add_argument("--fold", default='0', type=int) #Used for k-fold cross validation in tadpole/ukbb
    
    parser.add_argument("--conv_layers", default=[[32,32],[32,16],[16,8]], type=lambda x :eval(x))
    parser.add_argument("--dgm_layers", default= [[32,16,4],[],[]], type=lambda x :eval(x))
    parser.add_argument("--fc_layers", default=[8,8,1], type=lambda x :eval(x))
    parser.add_argument("--pre_fc", default=None, type=lambda x :eval(x))

    parser.add_argument("--gfun", default='gcn')
    parser.add_argument("--ffun", default='gcn')
    parser.add_argument("--k", default=3, type=int) 
    parser.add_argument("--pooling", default='add')
    parser.add_argument("--distance", default='euclidean')

    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--lr", default=5e-1, type=float)
    parser.add_argument("--test_eval", default=10, type=int)

    parser.set_defaults(default_root_path='./log')
    params = parser.parse_args(namespace=params)

    run_training_process(params)