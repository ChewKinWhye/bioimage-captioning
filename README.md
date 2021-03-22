<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Installation](#installation)
* [Usage](#usage)
* [Contact](#contact)


<!-- ABOUT THE PROJECT -->
## About The Project

Chest radiography is the most common imaging examination performed globally, used for screening and diagnosis of many critical illnesses in clinical practice2. The outcomes of radiography inform physicians of a patient’s medical condition, influencing healthcare provided. Typically, medical reports for radiographic images are written manually. This process is tedious and prone to human error. Additionally, because medical report generation requires expert domain knowledge, there is often a shortage of human operators. The generation of medical reports thus represents a bottleneck in patient healthcare. 

Our project aims to tackle the above problems by developing a model for automatic generation of medical X-ray reports. This will reduce load on clinicians and accelerate diagnostic processes. Furthermore, such a model also supports the development of explainable and trustworthy AI, where model outcomes can be explained and are understandable by humans.   


<!-- GETTING STARTED -->
## Getting Started

### Installation

1. Clone the repo
```sh
git clone https://github.com/ChewKinWhye/bioimage-captioning.git
```
2. Create venv
```sh
cd bioimage-captioning
python3 -m venv env
source env/bin/activate 
```
3. Install requirements
```sh
pip install -r requirements.txt
```
4. Install Kaggle dataset

Go to the 'Account' tab of your Kaggle user profile (https://www.kaggle.com/<username>/account) and select 'Create API Token'.
Open up downloaded kaggle.json file to obtain username and key.
```sh
export KAGGLE_USERNAME=username
export KAGGLE_KEY=key
kaggle datasets download -d raddar/chest-xrays-indiana-university -p data
unzip chest-xrays-indiana-university.zip
5. Install cheXpert datasta
At the bottom of the CheXpert webpage, write a registration form to download the CheXpert dataset. You will receive an email with the download link. Right-click your mouse on the download link(439GB or 11GB) and click 'Copy link address'
wget -P data -O chexpert "link_address_you_copied"
unzip chexpert
```

<!-- USAGE EXAMPLES -->
## Usage

For DNA modifications

```sh
python VAE_DNA.py --data_path /hdd/modifications/ecoli/deepsignal/ --data_size 500000 --rc_loss_scale 1 --output_filename VAE_DNA
```

For RNA modifications

```sh
python VAE_RNA.py --data_path /hdd/modifications/ecoli/deepsignal/ --data_size 900000 --rc_loss_scale 8 --output_filename VAE_RNA
```
More training options

```
Additional optional training parameters

  --data_path                 Path to data directory
  --output_filename           Name of the output file to save results
  --vae_epochs                Number of epochs to train VAE
  --vae_batch_size            Batch size of VAE training
  --predictor_epochs          Number of epochs to train VAE
  --predictor_batch_size      Batch size of predictor training
  --latent_dim                Latent dimension of VAE encoding space
  --data_size                 Size of dataset to use
  --rc_loss_scale             Scale value of the reconstruction error
```

<!-- CONTACT -->
## Contact

Chew Kin Whye - kinwhyechew@gmail.com

