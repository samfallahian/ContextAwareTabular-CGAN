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
`python main.py --train_type cae_gan --data_path dataset_test/adult/adult.csv --metadata_path dataset_test/adult/meta_data.json --dataset_name adult --labels "['year','month']"`  

For parallel computing:  
`parallel -q :::: commands.txt`

Run Screen: 
* `sudo apt-get install screen`
* `sudo apt-get install tmux`  

Start a new session:  
* `screen -S mysession`
* `tmux new -s mysession`  

You can detach from the session and leave it running in the background by pressing Ctrl-a d in screen or Ctrl-b d in tmux  
You can later reattach to the session with:  
* `screen -r mysession`
* `tmux attach -t mysession`

The contents of these log files can be followed in real-time using:  
* `tail -f output*.log`  

List all sessions with:  
* `screen -ls`
* `tmux ls`  

Once inside the session, you can terminate it by executing the exit command, or pressing Ctrl-D.  

Another way to kill a session without attaching to it:  
* `screen -X -S [session_name] quit`
* `tmux kill-session -t [session_name]`