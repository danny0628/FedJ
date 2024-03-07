# FedCAR: Cross-client Adaptive Re-weighting for Generative Models in Federated Learning


## Getting Started
### Prerequisites

What things you need to install the software and how to install them

```
conda create -n flgan python==3.8
conda activate flgan
conda install -y pytorch==1.7.1 torchvision torchaudio cudatoolkit=11.0 -c pytorch -c conda-forge
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
pip install -r requirements.txt

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


## Built With

* [Flower](https://github.com/adap/flower) - Federated Learning
* [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) - GAN
