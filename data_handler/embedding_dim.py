import pandas as pd


def cal_dim(df, categorical_columns, labels):
    X = df.drop(labels, axis=1)
    numerical_columns = [col for col in df.columns if col not in categorical_columns]
    categorical_columns = [col for col in categorical_columns if col not in labels]
    X_num = X[numerical_columns]
    cat_dims = [int(X[col].nunique()) for col in categorical_columns]
    # Define the embedding size per categorical column (rule of thumb: min(50, (size+1)//2))
    emb_dims = [(size, min(50, (size + 1) // 2)) for size in cat_dims]
    num_feature_no = X_num.shape[1]
    output_sizes = [df[label].nunique() for label in labels]
    print("output_sizes", output_sizes)

    return num_feature_no, emb_dims, output_sizes
