# <p style="text-align: center;">Context Aware Conditional GAN</p>
## <p style="text-align: center;">Tabular Data Generator</p>
### Overview
The generation of synthetic tabular data has numerous applications, ranging from data augmentation to privacy preservation. However, existing generative models often struggle to accurately capture the complex relationships and distributions present in real-world tabular datasets. To address this challenge, we propose a Context-aware Conditional Generative Adversarial Network (CA-CGAN) that leverages transfer learning to incorporate contextual information and improve the quality of the generated synthetic data. Specifically, we employ a pre-trained Contractive Autoencoder (CAE) to initialize the generator's input, enabling a more meaningful representation of the original data.
### Prerequisites
* Python 3.8
* torch 2.0.1
* pandas
* numpy
* scikit-learn
### Usage
Training:  
`python main.py --dataset=Adult --cae_model=MODEL_PATH --train`  
For parallel computing:  
`parallel -q :::: commands.txt`

