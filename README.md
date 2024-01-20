# Doubly Stochastic Variational Inference for Deep Gaussian Processes
+ New Environment Test
  
Succeeded with Mac M1 with TensorFlow=2.13, GPflow=2.2;

+ Trials

  -  whiten=True;
 
  -  Identity mean function at the final layer;
    
  - identity prior mean function for inducing inputs $\mathbf{Z}^{(l-1)}$ at the $l$-th intermediate layer, whose form is the same as the prior mean of outputs $\mathbf{F}^{(l)}: \mathbb{E}[\mathbf{F}^{(l)}] = m(\mathbf{F}^{(l-1)})=\mathbf{F}^{(l-1)}$.

+ from the forked: 
  
🤿 Implementation of doubly stochastic deep Gaussian Process using [GPflow 2.0](https://github.com/GPflow/GPflow) and [TensorFlow 2.0](https://github.com/tensorflow/tensorflow).

Heavily based on a previous implementation of [Doubly-Stochastic-DGP](https://github.com/ICL-SML/Doubly-Stochastic-DGP) and the [paper](https://arxiv.org/abs/1705.08933)

```
@inproceedings{salimbeni2017doubly, 
  title={Doubly stochastic variational inference for deep gaussian processes}, 
  author={Salimbeni, Hugh and Deisenroth, Marc}, 
  booktitle={Advances in Neural Information Processing Systems}, 
  year={2017} 
}
```

Includes demos for the step function and MNIST data set.
