# d2l PyTorch
This repo is used as the notebook during my learning experience of 
PyTorch and code in deep learning

The content in the repo is divided into three parts:
- d2l: code in the book [Dive into Deep Learning](https://d2l.ai/) and some practice programming by myself
- GNN: code to build up different types of GNNs, which lacks in the book d2l
- KG: code to construct *Knowledge Graph*

**Star or Fork this program if you like it**

### requirements
python==3.9

d2l==0.17.6

torch==1.12.1 + cu113

torchvision==0.13.1 + cu113

pandas==1.2.4

numpy==1.25.1

networkx==3.1

matplotlib==3.5.1

spacy==3.6.0

### running environment

IDE: PyCharm 2023

Environment: conda 4.9.2

## contents
```
|--d2l
|  |--chapter_2 : background knowledge
|  |  |--2.1_data_operation
|  |  |--2.2_data preprocess
|  |  |--2.3_linear_algebra
|  |  |--2.4_calculus
|  |  |--2.5_autograd
|  |  |--2.6_probability
|  |--chapter_3 : linear neural networks
|  |  |--3.1_linear_regression_preparation
|  |  |--3.2_linear_regression_implementation_from_scratch
|  |  |--3.3_linear_regression_implementation_by_pytorch
|  |  |--3.5_basic_of_the_FashionMNIST_dataset
|  |  |--3.6_softmax_regression_implementation_from_scratch
|  |  |--3.7_softmax_regression_implementation_by_pytorch
|  |--chapter_4 : multilayer perceptrons
|  |  |--4.1_activation_function
|  |  |--4.2_multilayer_perceptrons_implementation_from_scratch
|  |  |--4.3_multilayer_perceptrons_implementation_by_pytorch
|  |  |--4.4_model_selection_underfitting_overfitting
|  |  |--4.5_weight_decay
|  |  |--4.6 & 4.8_dropout & initialization
|  |  |--4.10_kaggle_house_price
|  |--chapter_5 : deep learning computation
|  |  |--5.1_block
|  |  |--5.3_parameter_management
|  |  |--5.4_custom_layer
|  |  |--5.5_IO
|  |  |--5.6_GPU
|  |--chapter_6 : convolutional neural networks
|  |  |--6.2_cross_correlation
|  |  |--6.3_padding_and_strides
|  |  |--6.4_multiple_input_and_output_channels
|  |  |--6.5_pooling
|  |  |--6.6_LeNet
|  |--chapter_7 : modern convolutional neural networks
|  |  |--7.1_AlexNet
|  |  |--7.2_VGG
|  |  |--7.3_NiN
|  |  |--7.4_GoogLeNet
|  |  |--7.5_batch_normalization
|  |  |--7.6_resnet
|  |  |--7.7_on_server
|  |--chapter_8 : recurrent neural networks
|  |  |--8.1_sequence
|  |  |--8.2_data_process
|  |  |--8.3_language_model
|  |  |--8.4_Basic_knowledge_of_RNN
|  |  |--8.5_RNN_from_scratch
|  |  |--8.6_RNN_by_pytorch
|  |--chapter_9 : modern recurrent neural networks
|  |  |--9.1_GRU
|  |  |--9.2_LSTM
|  |  |--9.3_deep_rnn
|  |  |--9.4_bidirectional_rnn
|  |  |--9.5_machine_translation
|  |  |--9.6_encoder_decoder
|  |  |--9.7_seq2seq
|  |--chapter_10 : attention mechanism
|  |  |--10.1_attention
|  |  |--10.2_Nadaraya_Waston_Kernel_regression
|  |  |--10.3_attention_scoring_function
|  |  |--10.4_Bahhdanau_attention
|  |  |--10.5_multihead_attention
|  |  |--10.6_self_attention
|  |  |--10.7_Transformer
|  |--chapter_11 : optimization algorithms
|  |  |--11.1_optimization
|  |  |--11.2_convex_optimization
|  |  |--11.3_gradient_descent
|  |  |--11.4_stochastic_gradient_descent
|
|--GNN
|
|--KG
|  |--demo1: build up a simple KG

```