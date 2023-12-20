# dataset_distillation
Implementation of the [dataset distillation algorithm from Dataset Distillation by Wang et al](https://arxiv.org/abs/1811.10959). The notebook is well-documented and easily extensible to different models/datasets. I implemented the algorithm using Tensorflow as well, however, it proved to be a huge headache to get working due to the clunkiness of TensorFlow's gradient tapes, and additional reasons outlined in the overview (which is why I opted to use Pytorch instead).

## Overview
For a fixed model and corresponding dataset, we generate a significantly smaller synthetic dataset such that the model can achieve similar results on the new dataset. Wang et al. outline the following algorithm for this:
<img width="940" alt="image" src="https://github.com/obround/dataset_distillation/assets/75817213/dfcac066-e699-4af9-be38-006c5ac78f46">

This algorithm was applied to the MNIST dataset. While implementing this algorithm, I ran into a major implementation issue: optimizers updates aren't differentiable in pytorch as required by the algorithm. To get around this issue, I used the [higher](https://github.com/facebookresearch/higher) library from Meta Research which provided differentiable optimizers.

## Results
My results were significantly limited by the computational power available to me (I don't have a GPU, so I had to run this on my CPU). Nevertheless, they perfectly illustrated the paper. After the distilled images were synthesized, the corresponding model was trained on them, yielding the following result on the final epoch:
```
Epoch 75
-------------------------------
loss: 0.470158  [    0/  100]
Test Error: 
 Accuracy: 37.5%, Avg loss: 2.058357 
```
The accuracy would have been 10% if the distilled data had no effect. The distilled images can be improved by:
 - Increasing the number of inner gradient descent steps (`gd_steps`, currently at `1` due to my limited computational resources)
 - Increasing the batch size (`BATCH_SIZE`, currently at `256`)

Making these two changes would likely lead to the results from the paper. Here is a picture of the distilled images (10 distilled images for each class):
<center>
<img src="https://github.com/obround/dataset_distillation/assets/75817213/7a82ccd1-5afb-4792-bc11-87ccc2072deb" alt="drawing" width="750"/>
</center>

Not very enlightening, eh? According to the paper, throwing more computational resources at this (using the methods I outlined previously) will make the digit's salient features to pop out more, like this (image is from the paper):
<img width="889" alt="image" src="https://github.com/obround/dataset_distillation/assets/75817213/a68bce96-6cf8-42df-96a0-3003ba285759">
