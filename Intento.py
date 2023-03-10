
import pickle
import numpy as np

import torch
from torch_geometric.data import Dataset, DataLoader, Data
from torch_geometric.transforms import NormalizeFeatures

with open('data/Datos/x_numerical.pkl', 'rb') as f:
    x_numeric = pickle.load(f)

with open('data/Datos/x_textual.pkl', 'rb') as f:
    x_text = pickle.load(f)

with open('data/Datos/y_.pkl', 'rb') as f:
    y = pickle.load(f)

y = np.expand_dims(y,axis = 2)

x = np.concatenate((x_numeric, x_text), axis = 2)
datos = np.concatenate((x,y), axis = 2)

def create_adj_matrix(num_nodes, density):
    # Create a random tensor of size (num_nodes, num_nodes)
    rand_tensor = torch.randn(num_nodes, num_nodes)
    
    # Threshold the tensor to obtain a binary adjacency matrix
    threshold = np.quantile(rand_tensor.numpy(), 1 - density) # calculate the threshold value based on the desired density
    adj_matrix = (rand_tensor >= threshold).to(torch.float32)
    adj_matrix.fill_diagonal_(0) # set the diagonal to 0
    
    return adj_matrix

matriz = create_adj_matrix(3, 0.05)

edge_index = matriz.nonzero().t()

data_list = []
for i in range(730):
    for j in range(198):
        data = {'x':x[i,j,:],'y':y[i,j,0],'edge_index':edge_index}
        data_list.append(data)
        print(i,j)

class MyDataset(Dataset):
    def __init__(self, data_list):
        super(MyDataset, self).__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def get(self, idx):
        data = self.data_list[idx]
        x = torch.tensor(data['x'], dtype=torch.float)
        y = torch.tensor([data['y']], dtype=torch.float)
        edge_index = torch.tensor(data['edge_index'], dtype=torch.long)
        edge_attr = torch.tensor(data['edge_attr'], dtype=torch.float)
        return Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

dataset = MyDataset(data_list)
loader = DataLoader(dataset, batch_size=1)

class datasetPropio():
    def __init__(datos, edge_index):
        self.X = datos.reshape((730 * 198, 14))
        self._is_protocol = False
        self.edge_index = edge_index
        self.mask = 
        self.n_features = 14
        self.num_classes = 1
        self.samples_per_epoch = 100
        self.y = datos[:, :, -1].flatten()