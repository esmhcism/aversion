# Risk Aversion: A Theoretical Exploration for Few-Shot Class-Incremental Learning

## Abstract
Few-Shot Class-Incremental Learning (FSCIL) aims to assimilate new concepts with minimal training samples while preserving the integrity of pre-established knowledge. Despite the efficacy of existing approaches, a critical yet underexplored aspect of FSCIL is the understanding of its intrinsic statistical principles, leading to the pivotal question: What are the key factors in addressing FSCIL? In response, we conduct a comprehensive theoretical exploration on FSCIL, focusing on the aversion of transfer and consistency risks. By tightening the upper bounds of both risks, we formulate practical guidelines for FSCIL implementation. These guidelines include expanding training datasets for base classes, optimizing classification margin discrepancy, preventing excessive focus on specific features, and ensuring unbiased classification across both base and novel classes. Following these insights, our experiments validate these principles, achieving state-of-the-art performance on three FSCIL benchmark datasets.

## Requirements
conda env create -f environment.yml

## Datasets
We follow [FSCIL](https://github.com/xyutao/fscil) setting to use the same data index_list for training.  
For CIFAR100, the dataset will be download automatically.  
For miniImagenet and CUB200, you can download from [here](https://drive.google.com/drive/folders/11LxZCQj2FRCs0JTsf_dafvTHqFn2yGSN?usp=sharing). Please put the downloaded file under `data/` folder and unzip it:
    
    $ tar -xvf miniimagenet.tar 
    $ tar -xvzf CUB_200_2011.tgz

## Checkpoints

You can download the checkpoints from [here](https://github.com/esmhcism/aversion.git).

## Testing scripts
cifar100
    $ python test.py -project base -dataset cifar100 -model_dir checkpoint/cifar.pth > output.txt

mini_imagenet
    $ python test.py -project base -dataset mini_imagenet -model_dir checkpoint/mini.pth > output.txt

cub200
    $ python test.py -project base -dataset cub200 -model_dir checkpoint/cub.pth > output.txt
