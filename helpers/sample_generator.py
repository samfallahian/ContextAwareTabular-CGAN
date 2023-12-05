from data_handler.data_reader import read_csv
from helpers.noise_generator_full import NoiseGenerator
from models.gan_model import Generator
from trainers.gan_sampler import CTGAN
import torch

if __name__ == '__main__':
    # discrete_columns = [
    #     'workclass',
    #     'education',
    #     'marital-status',
    #     'occupation',
    #     'relationship',
    #     'race',
    #     'sex',
    #     'native-country',
    #     'income'
    # ]

    discrete_columns = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'lroot_shell', 'is_guest_login',
                        'label']

    # real_data = read_csv("/mnt/d/sources/ca-cgan/ctgan/dataset/adult.csv")
    # input_size = 156
    real_data = read_csv("/mnt/d/sources/ca-cgan/ctgan/dataset/kddcup99e.csv")
    input_size = 261

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # cae_path = '/mnt/d/sources/ca-cgan/ctgan/saved_models/cae_final_saved_model_09262023.pth'
    cae_path = '/mnt/d/sources/ca-cgan/ctgan/saved_models/kdd/cae/cae_final_saved_model_kdd_10202023.pth'
    noise_generator = NoiseGenerator(model_path=cae_path, input_size=input_size, hidden_size=256, latent_size=64,
                                     device=device)

    # model_path = '/mnt/d/sources/ca-cgan/ctgan/saved_models/generator_cae_final_saved_model_09272023_cuda_latent_random_noise.pth'
    # model_name = "generator_cae_final_saved_model_09272023_cuda_full_normal_noise"
    # model_name = "generator_cae_final_saved_model_09272023_cuda_latent_normal_noise"
    model_name = "kdd_vanilla_gan"

    model_path = '/mnt/d/sources/ca-cgan/ctgan/saved_models/kdd/vanila_gan/generator_final_saved_model_10212023_cuda.pth'

    ctgan = CTGAN(saved_generator=model_path, verbose=True, save_directory='saved_models',
                  noise_generator=noise_generator,
                  device=device)
    ctgan.fit(real_data, discrete_columns)

    # model_path = '/Users/mfallahi/Sources/ca-cgan/ctgan/saved_models/cae_final_saved_model_09242023.pth'

    # generator.load_state_dict(torch.load(model_path))
    # generator.eval()

    # loaded_model = ctgan.load(model_path)
    #
    synthetic_data = ctgan.sample(10000)

    synthetic_data.to_csv(f"/mnt/d/sources/ca-cgan/ctgan/results/{model_name}.csv", index=False)

    print(synthetic_data.head(10))
