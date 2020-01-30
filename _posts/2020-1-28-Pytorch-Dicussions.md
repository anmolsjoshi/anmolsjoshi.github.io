---
layout: post
title: PyTorch Discussion Tips and Tricks
---

I did a deep dive of the discussion forum of PyTorch to find frequently asked questions, tips and tricks, and helpful 
techniques. If you haven't checked it out, here's a link - <https://discuss.pytorch.org>

It's an extremely well moderated forum, I'd really like to thank the users for their amazing contributions to 
this forum and helping novices like me get started with training neural networks.

## Exponential Moving Average

Implementing Polyak Averaging/Exponential Moving Average of model weights. Training neural networks can be a tricky
process and using Stochastic Gradient Descent can lead to erratic updates to model weights from different batches of
in your training dataset. To avoid this, it's helpful to create a copy of the model at the start of training, and
keep a moving average of the model weights to evaluate the model's performance. 

&theta;<sub>t</sub><sup>*</sup> = &alpha; &theta;<sub>t-1</sub> + (1-&alpha;) &theta;<sub>t</sub>

```python
class EMA:
    """Exponential moving average of model parameters.
    Args:
        model (torch.nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
    """
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        """Assign exponential moving average of parameter values to the
        respective parameters.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        """Restore original parameters to a model. That is, put back
        the values that were in each parameter at the last call to `assign`.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]

```

A popular example of using this is in training Bidirectional Attention Flow for Machine Comprehension <Page 6, Model Details>.

Assuming you have implemented the above handler, here is pseudo code below to show you how to use it. 

```python
ema = EMA(model, decay=0.999)

for epoch in range(num_epochs):
   for step, batch in enumerate(train_loader):
       model.train()
       optimizer.zero_grad()
       input, target = batch
       batch_size = input.shape[0]
       output = model(input)
       loss = criterion(output, target)
       loss.backward()
       optimizer.step()
       ema(model, step // batch_size)
    
   if validate_every % epoch == 0:
       ema.assign(model)
       metrics = evaluate_model(validation_loader)
       ema.resume(model)
``` 

Code is from: - 
* <https://github.com/chrischute/squad/blob/master/util.py#L174-L220>
* <https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py#L252-L532>
                 
## L1/L2 Regularization
Adding L2 regularization to training models. PyTorch's Optim package provides a wide variety of optimizers, all of them
come with a *weight_decay* parameter. This parameter adds a L2 regularization update to the gradients to all the model
weights. Let's examine the torch.optim.SGD source code. 

People might be more familiar with L2 regularization in the final loss calculation, 

loss = criterion(y_pred, y_true) + 0.5 * ||W||<sup>2</sup>

Below, you'll see that L2 regularization is not applied to loss, it's directly added to the gradient. To better understand
this, let's differentiate the above expression.

dW = dC/dW + W

PyTorch's inplace operation of add_ is used here, it simply takes the gradient of the model parameter and adds it to the product
of weight_decay and the value of the parameter. See docs here. 

```python
import torch
from .optimizer import Optimizer, required


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
    
        Considering the specific case of Momentum, the update can be written as
    
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
    
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
    
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
    
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
    
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:              # <--------------------------------
                    d_p.add_(weight_decay, p.data) # <--------------------------------
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss
``` 
    
Given that deep learning is an evolving field, with new variants of methods being published every day, it's 
helpful to know how to implement L1 and L2 regularization independently. All you need is your model, loss function,
and a regularization weight. 

Below is an example of how to avoid adding biases to L1/L2 regularization. Just to show the flexibility of PyTorch!

```python
def regularization(model, p):
   """
   p = 1 for L1 regularization
   p = 2 for L2 regularization
   """
   loss = torch.tensor([0.0], requires_grad=True)
   for name, parameter in model.parameters():
       if not 'bias' in name:
           loss += parameter.norm(p)
   return loss   

loss = loss_fn(y_pred, y) + lambda_reg * regularization(model, 2)
```

Note that we initialize loss as torch.FloatTensor that requires gradient. This is important to make sure Autograd registers
the effect of regualarization for desired backpropogation results. 


## Correct Usage of Loss Functions
Using incorrect inputs to calculate loss and metrics. We see this a lot in tutorials and I'm ashamed to admit it that 
I've been guilty of this too. This usually happens in classification problems where sigmoid or sotfmax is applied to the 
output of the last layer.

I caught this issue when I switched from Keras to PyTorch. Keras is a fantastic tool, where a user can build networks 
quickly and efficiently. But it's important to pay attention to the source code. [Look here!](https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/backend/numpy_backend.py#L333-L339)
Keras accounts for the user applying softmax to the output of the last layer.

```python
import torch.nn as nn

# Expects a sigmoid applied to the output of the last layer
loss = nn.BCELoss()

# As the name suggests, expects raw logits. 
loss = nn.BCEWithLogitsLoss()

# Expects raw logits.
loss = nn.CrossEntropyLoss()

# Expects log_softmax applied to output of last layer
loss = nn.NLLLoss()
```   

Although applying softmax results in probabilities of the network output, it is essentially unnecessary as taking the
max of the raw logits vs max of the softmax-ed logits gives the same answer. 

## Initialization
To prevent headaches in training your network, one must ensure correct initialization of weights at the start of training
process. Proper initialization of weights results in efficient gradient flow during backpropogation, ensures the distribution/variance
of the input is maintained in between layers, and no saturation of layer outputs. 

There has been extensive research on this subject, there are few papers that are considered standard and are used widely.

[Xavier Initialization](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) - Initialize weights with a distribution given below, works best for tanh, softplus activations.

[Kaiming He Initialization](https://arxiv.org/abs/1502.01852) - Works best for ReLu and its variants. Also, default for PyTorch, considered a reasonable default given popularity of Relu and its variants.

Both these initializations have normal and random distribution variants.

Below is code showing a custom initialization strategy based on layer type.

```python
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
```

- <https://discuss.pytorch.org/t/how-to-initialize-the-conv-layers-with-xavier-weights-initialization/8419>

## pad_sequence
**pad_sequence** is a helpful function that accepts a list of tensors and pads them along a certain dimension.
This is helpful in creating batches. See below. 

```python
import torch
from torch.nn.utils.rnn import pad_sequence

lengths = torch.tensor([2, 6, 4])
sentences = [torch.randint(0, 1000, size=(length,)) for length in lengths]
padded_sentences = pad_sequence(sentences, batch_first=True)

# tensor([[200, 259,   0,   0,   0,   0],
#         [399, 729, 497, 764, 365, 516],
#         [911, 528, 771, 238,   0,   0]])
```
3 tensors of different lengths are padded to the length of the longest tensor. Note the default padding value is 0.
If you specify batch_first, make sure that the rest of your calculations and transformations are consistent with that.

## pack_pad_sequence
**pack_pad_sequence** is a function that saves on comuptation of RNN. Let's assume that we're working with the batch
of sentences above. We pass it through an Embedding layer with the padding_idx specified as 0. 

```python
embedding = nn.Embedding(40000, 2, padding_idx=0)
max_length = padded_sentences.shape[1]
padded_sentences = embedding(padded_sentences)

# tensor([[[ 0.4036, -0.2818],
#          [-2.2939, -0.0033],
#          [ 0.0000,  0.0000],
#          [ 0.0000,  0.0000],
#          [ 0.0000,  0.0000],
#          [ 0.0000,  0.0000]],

#         [[ 0.6129, -0.1733],
#          [-1.2074,  1.7110],
#          [ 1.7345,  0.6253],
#          [ 0.2599,  1.7643],
#          [ 2.4085,  0.8773],
#          [-0.6243,  1.1937]],

#         [[ 0.5094,  0.2214],
#          [-0.0215, -1.0906],
#          [-0.2340, -1.9780],
#          [ 0.9354,  0.1259],
#          [ 0.0000,  0.0000],
#          [ 0.0000,  0.0000]]], grad_fn=<EmbeddingBackward>)
```  

padded_sentences is now of shape (3, 6, 2). Since we have 6 timesteps per batch, 18 total computations are needed for
an RNN; this is inefficient because sentences are padded and it is unnecessary to apply the RNN over those timesteps.
This is where pack_pad_sequence helps us, it extracts just the non-padded elements from the input and orders them to
effectively run the RNN over it. 

This function expects input ordered by length.

```python
lengths, sort_idx = lengths.sort(descending=True)
padded_sentences = padded_sentences[sort_idx]
padded_sentences = pack_padded_sequence(padded_sentences, lengths, batch_first=True)

# PackedSequence(data=tensor([[ 0.6129, -0.1733],
#         [ 0.5094,  0.2214],
#         [ 0.4036, -0.2818],
#         [-1.2074,  1.7110],
#         [-0.0215, -1.0906],
#         [-2.2939, -0.0033],
#         [ 1.7345,  0.6253],
#         [-0.2340, -1.9780],
#         [ 0.2599,  1.7643],
#         [ 0.9354,  0.1259],
#         [ 2.4085,  0.8773],
#         [-0.6243,  1.1937]], grad_fn=<PackPaddedSequenceBackward>), batch_sizes=tensor([3, 3, 2, 2, 1, 1]))
``` 

Closely examine the output of the embeddings and packed sequence, you'll see the first row is the first row of the longest
sentence, followed by the first input of the other inputs in decreasing lengths. Similarly the second row, and so on. 

You'll also notice batch_sizes, this creates batches of batch_sizes and passes it through the RNN. Where do these come from?
* first batch -> first row of each sentences
* second batch -> second row of each sentences (this completes the shortest sentence)
* third batch -> third row of the remaining two sentences

You get the idea! 

## pad_packed_sequence
**pad_packed_sequence** is a function to convert the packed sequences back to their padded form. Here you need to specify 
the length to pad the input to. The input is then padded with zero vectors (or value of your choosing). It's important
to sort the padded output back the original order, this is to maintain order with the target variable.

```python
lstm = nn.LSTM(2, 4, bias=True, num_layers=2)
padded_sentences, _ = lstm(padded_sentences)
padded_sentences, _ = pad_packed_sequence(padded_sentences, batch_first=True, total_length=max_length)
_, unsort_idx = sort_idx.sort(0)
padded_sentences = padded_sentences[unsort_idx]

# tensor([[[ 0.1253, -0.0174,  0.0477, -0.1064],
#          [ 0.2450, -0.0123,  0.1124, -0.1838],
#          [ 0.0000,  0.0000,  0.0000,  0.0000],
#          [ 0.0000,  0.0000,  0.0000,  0.0000],
#          [ 0.0000,  0.0000,  0.0000,  0.0000],
#          [ 0.0000,  0.0000,  0.0000,  0.0000]],

#         [[ 0.1219, -0.0176,  0.0426, -0.1075],
#          [ 0.2245, -0.0264,  0.0767, -0.2248],
#          [ 0.2663, -0.0149,  0.0777, -0.2684],
#          [ 0.2979, -0.0125,  0.0698, -0.3196],
#          [ 0.2905,  0.0016,  0.0683, -0.3235],
#          [ 0.3224,  0.0054,  0.0819, -0.3381]],

#         [[ 0.1203, -0.0196,  0.0391, -0.1194],
#          [ 0.2246, -0.0159,  0.0881, -0.1617],
#          [ 0.2961,  0.0020,  0.1277, -0.1718],
#          [ 0.3366,  0.0204,  0.1157, -0.1955],
#          [ 0.0000,  0.0000,  0.0000,  0.0000],
#          [ 0.0000,  0.0000,  0.0000,  0.0000]]], grad_fn=<IndexBackward>)
```

## Many to One LSTM
Many to One Problems. If you're unfamiliar with RNN's or this variant of classification problem, please check out
Andrej Karpathy's guide to RNN - it's one of the most helpful resources in understanding RNN's. Many to One simply
means that your input has many timesteps and output is just one item. An example of this is sentence classification, 
where you may have a sentiment label for each sentence, or an audio sample linked to a word.

To pass many samples through a model, we usually batch them. A batch must be a Tensor i.e. have a fixed shape. Not all
sentences will be the same length, we pad them sentences of a batch to the same length to process it easier. 

Here is the order of operations of a Many to One problem (sentence classification):
1. Sentence and Sentiment pairs are batched, sentences can be of varying length.
2. Sentences are padded to the maximum length of that batch.
3. Sentences are embedded using Word Embeddings (word2vec, glove, etc)
4. Embedded sentences are converted to a PackedSequence to effectively run input through RNN.
5. Embedded sentences are run through the RNN.
6. RNN output is converted back to padded output.
7. Hidden states from last timestep of each sentence is extracted. 
8. Linear layer with bias is applied to hidden states to produce raw logits. 

This example puts the previous three tips together!

```python

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SentimentClassifier(nn.Module):
    """
    Model to classify sentences
    
    Args:
        vocab_size (int): Size of vocabulary.
        embedding_size (int): Dimension of word vectors of Embedding Matrix.
        hidden_size (int): Size of hidden states of LSTM.
        num_classes (int): Number of classes.
        p (float): Probability of dropout. 
    """
    def __init__(self, 
                 vocab_size, 
                 embedding_size, 
                 hidden_size, 
                 num_classes, 
                 p):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(p)
        self.dense = nn.Linear(hidden_size, num_classes)
        
    
    def forward(self, x, lengths):
        """
        Args:
            x (Tensor): Batch of sentences, all padded to same length. Expected shape of (batch_size, max_length)
            lengths (Tensor): Length of sentences in batch. Expected shape (batch_size)
            
        Returns:
            logits (Tensor): Raw logits. Shape of (batch_size, num_classes)
        """
        batch_size, max_length = x.shape # (batch_size, max_length)
        
        # Embeds each word of sentence with word vectors
        x = self.embedding(x) # (batch_size, max_length, embedding_size)
        
        # Sorts batch by descending length, pack_padded_sequence requires this as input
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]
        x = pack_padded_sequence(x, lengths, batch_first=True) #PackedSequence

        # Encode embedded sentences using LSTM and unsort batch to original positions
        x, _ = self.lstm(x) 
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=max_length) # (batch_size, max_length, hidden_size)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]

        # Extract last timestep of encoded sentence, as above returns zeros for padded elements
        x = x[torch.arange(batch_size), lengths - 1] # (batch_size, hidden_size)
        
        # Applies Dropout to last timestep and applies a linear layer to extract logits
        x = self.dropout(x) # (batch_size, hidden_size)
        x = self.dense(x) # (batch_size, num_classes)
        return x
```

## Custom Loss Function
Creating custom loss functions is a question that comes up quite often, its as easy as defining a function, no need 
for nn.Module! As long as the inputs to the function have gradients enabled, you'll get PyTorch's autograd for free!
Here's an example of MSE Loss.

```python
def mse_loss(y, y_pred):
    assert y.shape == y_pred.shape
    batch_size = y.shape[0]
    return ((y_pred - y) ** 2).sum() / batch_size
```
    
## Multilabel Classification
A popular problem in machine learning is that of multilabel classification. What is it? It means each a sample of your data
has multiple labels related to it. For example, you have a dataset with red dresses, blue dresses, red cars and blue cars. 
You can set this is up as a multilabel problem where each sample will have a label of \['red/blue', 'car/dress']. How you handle this?

It's important to setup your DataLoader to output your features and labels as described above. When creating a model, have the output
of your model by raw logits. 

Options of loss functions are **MultiLabelSoftMarginLoss** and **BCEWithLogitsLoss**, note that these two will output the same result.
[See here](https://discuss.pytorch.org/t/what-is-the-difference-between-bcewithlogitsloss-and-multilabelsoftmarginloss/14944/12)

To calculate metrics, I'd like to shamelessly plug a library, I contribute to, called [Ignite](https://pytorch.org/ignite), it provides boilerplate code to help in
training and validating neural networks using PyTorch. 

Ignite offers accuracy, precision and recall with multilabel options, these work for a variety of input types and have been 
tested against scikit-learn's implementations of these metrics. 

```python
from ignite.metrics import Accuracy
acc = Accuracy(is_multilabel=True)
``` 

- <https://gist.github.com/bartolsthoorn/36c813a4becec1b260392f5353c8b7cc>


## Dropout Behaviour during Training and Testing
It's important to know the theory and inner workings of the models you create and their behaviour under different conditions. 
PyTorch implements inverted dropout, meaning that in train mode the inputs are masked and scaled by 1/p (p is Dropout Probability).
In eval mode, inputs pass through without scaling or masking. Why are inputs scaled during training? This is done to keep a similar
output from the dropout layer during training and evaluation.

```python
x = torch.FloatTensor([0, 1, 2, 3, 4, 5, 6, 7])
dropout = nn.Dropout(p=0.5)

dropout.train()
dropout(x)
# tensor([ 0.,  2.,  4.,  0.,  0.,  0., 12., 14.])

dropout.eval()
dropout(x)
# tensor([0., 1., 2., 3., 4., 5., 6., 7.])
``` 

During training mode, you'll see that some inputs are masked and the rest are multiplied by 2 (1/0.5 = 2).

## Batch Normalization - Running Metrics - Training and Testing

Batch Normalization was one of the most integral additions to the field of deep learning over the last few years. It provides
protection against poor initialization schemes and bad activation choices. It is important to understand batch normalization
before using it.

Batch Normalization on a high level tries to maintain the distributions across batches of input and layers of the model,
by keep a running mean and variance of layer outputs and using them to normalize the outputs as inputs to following layers.
Do you know how these metrics are calculated and stored?

Let's use an example of BatchNorm1d.

One thing to know is that batch normalization modules initialize running_mean with 0's the size of num_features and 1's
for running_var. This is done because Batch Normalization aims to maintain the batch distribution with mean=0, variance=1.

In this example, I'll pass 3 batches of (4, 4) through a batch normalization layer, keeping track of the mean and variance, 
and we'll compare with the batchnorm running metrics. 

Unless specified, the default exponential_average_factor will equal the initialized momentum parameter.
```python
batchnorm = nn.BatchNorm1d(num_features=4)
running_mean = torch.zeros(4)
running_var = torch.zeros(4)
exponential_average_factor = 0.1    # Default momentum of BatchNorm1d

batchnorm.train()
for i in range(3):
    x = torch.rand(4, 4)
    running_mean = exponential_average_factor * x.mean(dim=0) + (1 - exponential_average_factor) * running_mean
    running_var = exponential_average_factor * x.var(dim=0) + (1 - exponential_average_factor) * running_var
    batchnorm(x);

assert torch.allclose(batchnorm.running_mean, running_mean)
assert torch.allclose(batchnorm.running_var, running_var)

print (batchnorm.running_mean)
# tensor([0.1200, 0.1560, 0.1185, 0.1099])

print (batchnorm.running_var)
# tensor([0.7460, 0.7557, 0.7471, 0.7420])
``` 

Continuing the above example, Batch Normalization behaves differently during training and evaluation. During evaulation, 
running metrics are no longer calculated, and they are considered constants when normalizing inputs. In the example below,
we can see that running_mean and running_var do not change with new inputs in eval mode.

```python
batchnorm.eval()
for i in range(3):
    x = torch.rand(4, 4)
    batchnorm(x);

print (batchnorm.running_mean)
# tensor([0.1200, 0.1560, 0.1185, 0.1099])

print (batchnorm.running_var)
# tensor([0.7460, 0.7557, 0.7471, 0.7420])
```

## Proper Method of Running Inference
Not much information to add here, from the last two points - it's important to turn eval mode of the model to ensure the 
the layers that have a train and eval mode are using the correct mode. 

Additionally, its important to use `with torch.no_grad():` over the validation processing to prevent calculations of gradients
during inference. This results in speed-ups by removing unnecessary computations.

## How to Clip Gradients
From the discussion forums, I found two ways for performing gradient clipping in PyTorch. 

First option is to use **torch.nn.utils.clip_grad_norm_**, it clips the norm of the gradients as if they were a single vector
and modifies them accordingly.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
```

Another option is to use **torch.nn.utils.clip_grad_value_**, this clips the value of the gradients between the negative and
positive of the provided clip_value.

```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
```
It is important to note that these two methods clip the gradients after they are calculated.

But what if you wanted to clip the gradients during backpropogation, so clipped gradients flow through timesteps during backpropogation.
You can use backward hooks provided in PyTorch.

```python
def clip_grad(v, min, max):
    v_tmp = v.expand_as(v)
    v_tmp.register_hook(lambda g: g.clamp(min, max))
    return v_tmp
```

There is an important distinction to make between the two methods.

- <https://arxiv.org/abs/1308.0850>
- <https://discuss.pytorch.org/t/proper-way-to-do-gradient-clipping/191/22>
- <https://github.com/t-vi/pytorch-tvmisc/blob/master/misc/graves_handwriting_generation.ipynb>

## Batch Normalization, bias=False

Although this isn't a big deal, I noticed this in a few tutorials and decided to finally read the Batch Normalization paper. 
When using Batch Normalization, the bias from preceeding layer is set to False. This is because BatchNorm layer automatically 
adds a bias term, and its order of operations yields the preceeding layer's bias unchanged and unoptimized.

This is because batch normalization subtracts the mean of the batch from the input. If a constant is added to the input, 
the difference of the input and its mean will be the same as if the constant was not added.

See the simple example below:

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7])
print(x - np.mean(x))
# [-3. -2. -1.  0.  1.  2.  3.]

x += 1
print(x - np.mean(x))
# [-3. -2. -1.  0.  1.  2.  3.]
```

Let's see this in action. In the example below, I set up two models, only difference being bias flag in the linear layer. For both models, 
I ensure that all the parameters of the model are the same. 

Given the same input, same optimization methods, we optimize the model for 1000 steps. 

At the end, we see that linear weight of the layers are the same showing that having a bias had no effect on the optimization.
Another observation is that bias of the model is unchanged even after 1000 iterations.

This further proves that bias of preceeding layer has no effect when using Batch normalization.

```python
import torch
import torch.nn as nn
from torch.optim import SGD


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.linear = nn.Linear(4, 8)
        self.batch_norm = nn.BatchNorm1d(8)
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.fc(x)
        return x


class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.linear = nn.Linear(4, 8, bias=False)
        self.batch_norm = nn.BatchNorm1d(8)
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.fc(x)
        return x
    
model1 = Model1()
model2 = Model2()
model2.linear.weight = model1.linear.weight
model2.fc.weight = model1.fc.weight
model2.fc.bias = model1.fc.bias

initial_bias = model1.linear.bias
initial_bias
# Parameter containing:
# tensor([ 0.3368, -0.1422,  0.1352,  0.4685, -0.1898, -0.3018,  0.2751,  0.0719],
#        requires_grad=True)

x = torch.rand(2, 4)
y = torch.rand(2, 1)
criterion = nn.MSELoss()
optimizer1 = SGD(model1.parameters(), lr=1e-3)
optimizer2 = SGD(model2.parameters(), lr=1e-3)

for _ in range(1000):
    model1.zero_grad()
    model2.zero_grad()

    y1 = model1(x)
    y2 = model2(x)

    loss1 = criterion(y1, y)
    loss2 = criterion(y2, y)

    loss1.backward()
    loss2.backward()

    optimizer1.step()
    optimizer2.step()

model1.linear.weight

# Parameter containing:
# tensor([[ 0.0820, -0.3595, -0.4871, -0.3785],
#         [-0.1660,  0.1733,  0.3810, -0.1020],
#         [-0.4860, -0.2699, -0.1235,  0.3905],
#         [ 0.3131, -0.1860, -0.1112, -0.2838],
#         [-0.3188, -0.2247, -0.2293, -0.3067],
#         [ 0.2633, -0.4191, -0.0386, -0.4332],
#         [-0.1564,  0.4842,  0.1528, -0.3146],
#         [-0.0523,  0.2751, -0.4138, -0.3792]], requires_grad=True)

model2.linear.weight
# Parameter containing:
# tensor([[ 0.0820, -0.3595, -0.4871, -0.3785],
#         [-0.1660,  0.1733,  0.3810, -0.1020],
#         [-0.4860, -0.2699, -0.1235,  0.3905],
#         [ 0.3131, -0.1860, -0.1112, -0.2838],
#         [-0.3188, -0.2247, -0.2293, -0.3067],
#         [ 0.2633, -0.4191, -0.0386, -0.4332],
#         [-0.1564,  0.4842,  0.1528, -0.3146],
#         [-0.0523,  0.2751, -0.4138, -0.3792]], requires_grad=True)

torch.allclose(model1.linear.bias, initial_bias)
# True
```

## Multiple Loss Functions with Same Input

Below we setup our code with two loss functions, one that calculates the sum of difference of squares of the input and output.
The other does the same except that the input is raised to the fourth power. 

We calculate the gradient separately and can confirm that for the first loss function, the gradient wrt input should be 2 times the input.
For the second function, it is 4 times the cube of the input.

Finally, we see that PyTorch's autograd respects chain rule as the gradients wrt x are the sum of the gradients calculated separately.

```python
import torch
import torch.nn as nn

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y_true = torch.tensor([0.0, 0.0, 0.0])

def loss1(x, y_true):
    y = x ** 2
    return torch.abs(y - y_true).sum()


def loss2(x, y_true):
    y = x ** 4
    return torch.abs(y - y_true).sum()


loss = loss1(x, y_true)
loss.backward()
x.grad
# tensor([2., 4., 6.])

x.grad.data.zero_()
loss = loss2(x, y_true)
loss.backward()
x.grad
# tensor([  4.,  32., 108.])

x.grad.data.zero_()
loss = loss1(x, y_true) + loss2(x, y_true)
loss.backward()
x.grad
# tensor([  6.,  36., 114.])
```


## Added hook to debug NaNs in gradients
```python
import numpy as np
import torch
import torch.nn as nn

def create_hook(name):
    def hook(grad):
        print(name, grad)
    return hook

x = torch.tensor([1.0, np.nan])
k = nn.Parameter(0.01*torch.randn(1))

between = k.repeat(2) # need the extra dimensions in order to fix the NaN gradient
between.register_hook(create_hook('between'))

y = between*x

masked = y[:-1]

loss = masked.sum()
print(f'loss: {loss}')

loss.backward()
# loss: -0.01673412136733532
# between tensor([1., nan])
```
- <https://github.com/pytorch/pytorch/issues/15131#issuecomment-447149154>
