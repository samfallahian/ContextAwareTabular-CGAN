import torch
import torch.nn as nn


# class ClassifierModel(nn.Module):
#     def __init__(self, emb_dims, no_of_num):
#         super(ClassifierModel, self).__init__()
#         self.emb_layers = EntityEmbedding(emb_dims)
#         self.no_of_cat = sum([emb_dim for _, emb_dim in emb_dims])
#         self.no_of_num = no_of_num
#
#         # Define additional layers
#         self.lin1 = nn.Linear(self.no_of_cat + no_of_num, 200)
#         self.lin2 = nn.Linear(200, 100)
#         self.lin3 = nn.Linear(100, 50)
#         self.out = nn.Linear(50, 1)
#         self.bn1 = nn.BatchNorm1d(200)
#         self.bn2 = nn.BatchNorm1d(100)
#         self.bn3 = nn.BatchNorm1d(50)
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x_cat, x_num):
#         x_cat = self.emb_layers(x_cat)
#         x = torch.cat([x_cat, x_num], 1)
#         x = self.lin1(x)
#         x = self.bn1(x)
#         x = torch.relu(x)
#         x = self.lin2(x)
#         x = self.bn2(x)
#         x = torch.relu(x)
#         x = self.lin3(x)
#         x = self.bn3(x)
#         x = torch.relu(x)
#         x = self.out(x)
#         return x
class ClassifierModel(nn.Module):
    def __init__(self, emb_dims, no_of_num, output_sizes):
        super(ClassifierModel, self).__init__()
        self.emb_layers = EntityEmbedding(emb_dims)
        self.no_of_cat = sum([emb_dim for _, emb_dim in emb_dims])
        self.no_of_num = no_of_num
        # Define additional layers
        self.lin1 = nn.Linear(self.no_of_cat + no_of_num, 200)
        self.lin2 = nn.Linear(200, 100)
        self.lin3 = nn.Linear(100, 50)

        self.outs = nn.ModuleList([nn.Linear(50, size) for size in output_sizes])

        self.bn1 = nn.BatchNorm1d(200)
        self.bn2 = nn.BatchNorm1d(100)
        self.bn3 = nn.BatchNorm1d(50)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x_cat, x_num):
        x_cat = self.emb_layers(x_cat)
        x = torch.cat([x_cat, x_num], 1)
        x = self.lin1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.lin3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        outs = [out(x) for out in self.outs]
        return outs

class EntityEmbedding(nn.Module):
    def __init__(self, emb_dims):
        super(EntityEmbedding, self).__init__()
        # Create an embedding layer for each pair in emb_dims
        self.embeddings = nn.ModuleList([nn.Embedding(cat_dim, emb_dim) for cat_dim, emb_dim in emb_dims])

    def forward(self, x_cat):
        # Apply each embedding layer to the corresponding categorical feature
        x_cat = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_cat = torch.cat(x_cat, 1)
        return x_cat
