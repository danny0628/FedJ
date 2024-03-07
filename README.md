# FedCAR: Cross-client Adaptive Re-weighting for Generative Models in Federated Learning

## Updating..
## Getting Started
### Prerequisites

What things you need to install the software and how to install them

```
conda create -n fedcar python==3.8
conda activate fedcar
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
or Get inspired by your GPU experience : https://pytorch.org/get-started/previous-versions
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
pip install -r requirements.txt
```
## Train

Data format for stylegan2

```
python3 dataset_tool.py --source={your original dataset path} --dest={output dir path} --width={image size} --height={image_size}
```

For conditional training we need label json file

```
python3 make_labels.py --input_folder={stylegan2 format dataset path} --output={same input_folder path} --task='{normal or abnormal}'
```

running Training script
```
CUDA_VISIBLE_DEVICES={gpu_id} train.py --outdir './' --data '{dataset_path}' --batch={batch_size} --cond=True
```

## Built With

* [Flower](https://github.com/adap/flower) - Federated Learning
* [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) - GAN


