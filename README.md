# DualNet-ENE
Extranodal Extension (ENE) Identification on Computed Tomography with Deep Learning for Head and Neck Cancers
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a> [![arXiv](https://img.shields.io/badge/arXiv-2111.10480-b31b1b.svg)](https://arxiv.org/abs/2110.08424)

Keywords: Deep Learning, Convolutional Neural Network, CT, Head and Neck, Cancer, extranodal extension

This work is under review, and citation will be available upon publication.

## Introduction
Extranodal extension (ENE) occurs when tumor infiltrates through the lymph node capsule in the surrounding tissue. In head and neck cancer, ENE is an important factor for prognostication and treatment decision-making for head and neck cancer. ENE can only be diagnosed on surgical pathology, and it is very difficult for radiologists to predict ENE based on CT scans. DualNet-ENE is a deep learning, 3D-CNN model that has been trained on multi-institutional datasets of pathologically-annotated lymph nodes to accurately and reliably predict nodal metastasis and ENE ([Kann et al, Scientific Reports, 2018](https://www.nature.com/articles/s41598-018-32441-y), [Kann et al, Journal of Clinical Oncology, 2020](https://pubmed.ncbi.nlm.nih.gov/31815574/)). DualNet-ENE accepts a head and neck CT scan and a manually segmented lymph node mask as input. It outputs the probability of extranodal extension and nodal metastasis at the lymph node-level.

## Repository Structure
The DualNet-ENE repository is structured as follows:

* All the source code to reproduce the deep-learning-based pipeline is found under the `src` folder.
* Upon manuscript publication (currently under review), the model will be uploaded and hosted open-access on www.modelhub.ai  
* Five sample subjects' CT data and the associated manual node segmentation masks as well as all the models weights necessary to run the pipeline will be included for experimental pipeline testing

# Setup
This code was developed and tested using Python 3.8.5 on Ubuntu 20.04 with Cuda 11.2 and Tensorflow version 2.4

For the code to run as intended, all the packages under `requirements.txt` should be installed. In order not to break previous installations and ensure full compatibility, it's highly recommended to create a virtual environment to run the DeepCAC pipeline in. Here follows an example of set-up using `python virtualenv`:

```
# install python's virtualenv
sudo pip install virtualenv

# parse the path to the python3 interpreter
export PY3PATH=$(which python3)

# create a virtualenv with such python3 interpreter named "venv"
# (common name, already found in .gitignore)
virtualenv -p $PY3PATH venv 

# activate the virtualenv
source venv/bin/activate
```

At this point, `(venv)` should be displayed at the start of each bash line. Furthermore, the command `which python3` should return a path similar to `/path/to/folder/venv/bin/python3`. Once the virtual environment is activated:

```
# once the virtualenv is activated, install the dependencies
pip install -r requirements.txt
```

At this stage, everything should be ready for the data to be processed by the pipeline. Additional details can be found in the markdown file under `src`.

The virtual environment can be deactivated by running:

```
deactivate
```

# Disclaimer
The code and data of this repository are provided to promote reproducible research. They are not intended for clinical care or commercial use.

The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.