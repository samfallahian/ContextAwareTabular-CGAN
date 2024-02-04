import pandas as pd


def cal_dim(df, categorical_columns, label):
    X = df.drop(label, axis=1)
    numerical_columns = [col for col in df.columns if col not in categorical_columns]
    # categorical_columns.remove(label) # single label
    categorical_columns = [col for col in categorical_columns if col not in label]
    X_num = X[numerical_columns]
    # Calculate the number of unique values per categorical column
    cat_dims = [int(X[col].nunique()) for col in categorical_columns]
    # Define the embedding size per categorical column (rule of thumb: min(50, (size+1)//2))
    emb_dims = [(size, min(50, (size + 1) // 2)) for size in cat_dims]
    num_feature_no = X_num.shape[1]
    return num_feature_no, emb_dims
