import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_add_pool, global_max_pool, global_mean_pool
import torch.nn.functional as F
from torch_geometric import nn as gnn


class SubGcn(nn.Module):
    def __init__(self, c_in, hidden_size, nc):
        super().__init__()
        self.gcn = gnn.SGConv(c_in, hidden_size, K=3)
        # 1
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(hidden_size // 2, nc)
        )

    def forward(self, graph):
        h = F.relu(self.gcn(graph.x, graph.edge_index))
        h_avg = gnn.global_mean_pool(h, graph.batch)
        logits = self.classifier(h_avg)
        return logits


# Internal graph convolution feature module
class SubGcnFeature(nn.Module):
    def __init__(self, c_in, hidden_size):
        super().__init__()
        self.gcn = gnn.SGConv(c_in, hidden_size, K=3)

    def forward(self, graph):
        h = F.relu(self.gcn(graph.x, graph.edge_index))
        h_avg = gnn.global_mean_pool(h, graph.batch)
        return h_avg


class GraphNet(nn.Module):
    def __init__(self, c_in, hidden_size, nc):
        super().__init__()
        self.bn_0 = nn.LayerNorm(c_in)
        self.gcn_1 = gnn.GraphConv(c_in, hidden_size)
        self.bn_1 = nn.LayerNorm(hidden_size)
        self.gcn_2 = gnn.GraphConv(hidden_size, hidden_size)
        self.bn_2 = nn.LayerNorm(hidden_size)
        self.gcn_3 = gnn.GraphConv(hidden_size, hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 2),
            # nn.Dropout(),
            nn.Linear(hidden_size // 2, nc)
        )

    def forward(self, graph):
        # x_normalization = graph.x
        # h = F.relu(self.gcn_1(x_normalization, graph.edge_index))
        # h = F.relu(self.gcn_2(h, graph.edge_index))
        x_normalization = self.bn_0(graph.x)
        x_normalization = F.relu(x_normalization)
        h = self.bn_1(F.relu(self.gcn_1(x_normalization, graph.edge_index)))
        h = self.bn_2(F.relu(self.gcn_2(h, graph.edge_index)))
        h = self.bn_3(F.relu(self.gcn_3(h, graph.edge_index)))
        # h = F.relu(self.gcn_3(h, graph.edge_index))
        # logits = self.classifier(h + x_normalization)
        logits = self.classifier(h)
        logits = F.log_softmax(logits, dim=1)
        return logits, h


class GCNSimple(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCNSimple, self).__init__()
        self.norm = nn.BatchNorm1d(num_features)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.norm(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCN(nn.Module):
    """Flexible GCN network."""

    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hparams

        for param_name in [
            "num_node_features",
            "num_conv_layers",
            "conv_size",
            "lin1_size",
            "lin2_size",
            "output_size",
        ]:
            if not isinstance(hparams[param_name], int):
                raise Exception("Wrong hyperparameter type!")

        if hparams["num_conv_layers"] < 1:
            raise Exception("Invalid number of layers!")

        if hparams["activation"] == "relu":
            activation = nn.ReLU
        elif hparams["activation"] == "prelu":
            activation = nn.PReLU
        else:
            raise Exception("Invalid activation function name.")

        if hparams["pool_method"] == "add":
            self.pooling_method = global_add_pool
        elif hparams["pool_method"] == "mean":
            self.pooling_method = global_mean_pool
        elif hparams["pool_method"] == "max":
            self.pooling_method = global_max_pool
        else:
            raise Exception("Invalid pooling method name")

        self.conv_modules = nn.ModuleList()
        self.activ_modules = nn.ModuleList()

        self.conv_modules.append(GCNConv(hparams["num_node_features"], hparams["conv_size"]))
        self.activ_modules.append(activation())

        for _ in range(hparams["num_conv_layers"] - 1):
            self.conv_modules.append(GCNConv(hparams["conv_size"], hparams["conv_size"]))
            self.activ_modules.append(activation())

        self.lin1 = nn.Linear(hparams["conv_size"], hparams["lin1_size"])
        self.activ_lin1 = activation()

        self.lin2 = nn.Linear(hparams["lin1_size"], hparams["lin2_size"])
        self.activ_lin2 = activation()

        self.output = nn.Linear(hparams["lin2_size"], hparams["output_size"])

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for layer, activation in zip(self.conv_modules, self.activ_modules):
            x = layer(x, edge_index)
            x = activation(x)

        # x = self.pooling_method(x, batch)

        x = self.lin1(x)
        x = self.activ_lin1(x)

        x = self.lin2(x)
        x = self.activ_lin2(x)
        x = self.output(x)
        x = nn.Softmax(dim=1)(x)
        return x
