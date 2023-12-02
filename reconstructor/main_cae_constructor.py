"""CLI."""

from ctgan.data_handler.data_reader import read_csv
from ctgan.reconstructor.cae_reconstructor import CAEReconstructor
import torch


if __name__ == '__main__':
    discrete_columns = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country',
        'income'
    ]

    real_data = read_csv("../dataset/adult.csv")


    # device = torch.device(
    #     'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = '/ctgan/saved_models/cae_final_saved_model_09242023.pth'
    # model.load_state_dict(torch.load('/mnt/d/sources/cgan/saved_models/transformer_final_saved_model_09162023.pth'))

    cae = CAEReconstructor(model_path=model_path,verbose=True, device=device)
    input_data = real_data[:5]
    print(input_data)
    outputs = cae.fit(real_data, input_data , discrete_columns)


    # Decode the data using trained model
    # model.eval()
    # sample_latent = torch.randn(1, 64)
    # decoded_data = model.decode(sample_latent)
    # print(decoded_data.shape)

    # main()