import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


class ClassifierDataTransformer(nn.Module):
    def __init__(self, categorical_columns, label):
        super(ClassifierDataTransformer, self).__init__()
        # self.categorical_columns = categorical_columns
        self.categorical_columns = [col for col in categorical_columns if col not in label]
        self.label = label

    def forward(self, data):
        # Split features and target label
        X = data.drop(self.label, axis=1)
        y = data[self.label]
        # numerical_columns = [col for col in data.columns if col not in self.categorical_columns + [self.label]]
        numerical_columns = [col for col in data.columns if col not in self.categorical_columns + self.label]
        # Separate categorical and numerical data
        X_num = X[numerical_columns]

        # Initialize label encoders for each categorical column
        label_encoders = {}
        for col in self.categorical_columns:
            label_encoders[col] = LabelEncoder()
            X[col] = label_encoders[col].fit_transform(X[col])

        # Convert categorical columns to tensors
        X_cat = torch.tensor(X[self.categorical_columns].values, dtype=torch.int64)

        # Convert numerical columns to tensor
        X_num = torch.tensor(X_num.values, dtype=torch.float32)

        # Convert the target variable to a tensor
        label_encoder = LabelEncoder()
        # y = label_encoder.fit_transform(y.ravel())
        y = label_encoder.fit_transform(y.values.ravel())
        y = torch.tensor(y, dtype=torch.float32)
        y = y.unsqueeze(1)  # Add an extra dimension

        # Create TensorDatasets for training and testing data
        dataset = TensorDataset(X_cat, X_num, y)

        data_loader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=True)

        return data_loader
