import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv, global_mean_pool, global_add_pool
#----------------------------------------------------------------------------
                            # 3D Attention Network
#----------------------------------------------------------------------------

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)




class GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, layer='graphsage'):
        super(GCNEncoder, self).__init__()
        if layer == "gcn":
            # linear->GCN layer=2
            self.lin = nn.Linear(in_channels, out_channels)
            self.dropout1 = nn.Dropout()
            self.conv1 = GCNConv(out_channels, out_channels, improved=True)
            self.dropout2 = nn.Dropout()
            self.conv2 = GCNConv(out_channels, out_channels, improved=True)
        elif layer == "graphsage":
            # GraphSAGE
            self.lin = nn.Linear(in_channels, out_channels)
            self.dropout1 = nn.Dropout()
            self.conv1 = SAGEConv(out_channels, out_channels)
            self.dropout2 = nn.Dropout()
            self.conv2 = SAGEConv(out_channels, out_channels)
        elif layer == "gat":
            # GAT
            self.lin = nn.Linear(in_channels, out_channels)
            self.dropout1 = nn.Dropout()
            self.conv1 = GATConv(out_channels, out_channels, heads=1)
            self.dropout2 = nn.Dropout()
            self.conv2 = GATConv(out_channels, out_channels, heads=1)
        elif layer == "gin":
            # GIN
            self.lin = nn.Linear(in_channels, out_channels)
            self.dropout1 = nn.Dropout()
            self.conv1 = GINConv(
                nn.Sequential(
                    nn.Linear(out_channels, out_channels),
                    nn.ReLU(),
                    nn.Linear(out_channels, out_channels)
                )
            )
            self.dropout2 = nn.Dropout()
            self.conv2 = GINConv(
                nn.Sequential(
                    nn.Linear(out_channels, out_channels),
                    nn.ReLU(),
                    nn.Linear(out_channels, out_channels)
                )
            )

    def forward(self, x, edge_index):
        out = self.lin(x)
        out = self.dropout1(out)
        identity1 = out
        out1 = self.conv1(out, edge_index).relu()
        out1 += identity1
        out = self.dropout2(out1)
        identity2 = out
        out2 = self.conv2(out, edge_index)
        out2 += identity2
        return [out1, out2]


class GCNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, nclass):
        super(GCNClassifier, self).__init__()
        # self.ncov = ncov
        self.gcnencoder = GCNEncoder(in_channels, hidden_channels)
        self.dropout = nn.Dropout()
        self.linear1 = nn.Linear(25, 16)
        self.linear3 = nn.Linear(hidden_channels, nclass)

    def forward(self, x, edge_index, batch):
        # x = F.relu(self.linear1(x))
        outs = self.gcnencoder(x, edge_index)

        out = global_mean_pool(outs[-1], batch)
        #         out = global_add_pool(outs[-1], batch)
        #         out = torch.hstack([global_mean_pool(out, batch) for out in outs])

        # out = self.dropout(out)
        # out = self.linear1(out).relu()
        # # out = torch.cat((out, cov.view(-1, self.ncov)), dim=1)
        # out = self.linear2(out)
        # out = self.linear3(out)
        return out


class NFCS_Channel(nn.Module):
    def __init__(self):
        super(NFCS_Channel, self).__init__()
        self.dense1 = nn.Linear(4005,1024)
        self.relu = nn.ReLU(inplace=True)
        self.dense2 = nn.Linear(1024,128)
        self.dense3 =nn.Linear(128+64,96)
        self.droup =nn.Dropout(0.5)
        self.GCN = GCNClassifier(in_channels=25, hidden_channels=64, nclass=2)
    def forward(self, x,graphx,graphedge_index,graphbatch):
        gcnout = self.GCN(graphx, graphedge_index, graphbatch)
        x=x.squeeze(1)
        x=self.dense1(x)
        x=self.relu(x)
        x= self.dense2(x)
        x=self.relu(x)
        x = self.droup(x)
        out = torch.cat([x, gcnout], dim=1)
        x=self.dense3(out)
        return x



#----------------------------------------------------------------------------
                            # Define networks
#----------------------------------------------------------------------------


def nfsc_channel():
    model = NFCS_Channel()
    print_network(model)
    return model
