Dual Space Gradient Descent
===========================

This Python code implements Dual Space Gradient Descent (DualSGD), presented in the paper "Dual Space Gradient Descent for Online Learning" accepted at the 29th Conference on Neural Information Processing Systems (NIPS 2016).

The code is tested on Windows-based operating system with Python 2.7. Please make sure that you have installed *python-numpy* and *sklearn* to run the example.

Run the demo using this command
-------------------------------------
	python run_dsgd.py

Citation
--------

```
@InProceedings{trung_etal_nips16_dualsgd,
  author    = {Trung Le and Tu Dinh Nguyen and Vu Nguyen and Dinh Phung},
  title     = {Dual Space Gradient Descent for Online Learning},
  booktitle = {Advances in Neural Information Processing Systems 29 (NIPS)},
  year      = {2016},
  editor    = {D. D. Lee and M. Sugiyama and U. V. Luxburg and I. Guyon and R. Garnett},
  pages     = {4583--4591},
  publisher = {Curran Associates, Inc.},
  abstract  = {One crucial goal in kernel online learning is to bound the model size. Common approaches employ budget maintenance procedures to restrict the model sizes using removal, projection, or merging strategies. Although projection and merging, in the literature, are known to be the most effective strategies, they demand extensive computation whilst removal strategy fails to retain information of the removed vectors. An alternative way to address the model size problem is to apply random features to approximate the kernel function. This allows the model to be maintained directly in the random feature space, hence effectively resolve the curse of kernelization. However, this approach still suffers from a serious shortcoming as it needs to use a high dimensional random feature space to achieve a sufficiently accurate kernel approximation. Consequently, it leads to a significant increase in the computational cost. To address all of these aforementioned challenges, we present in this paper the Dual Space Gradient Descent (DualSGD), a novel framework that utilizes random features as an auxiliary space to maintain information from data points removed during budget maintenance. Consequently, our approach permits the budget to be maintained in a simple, direct and elegant way while simultaneously mitigating the impact of the dimensionality issue on learning performance. We further provide convergence analysis and extensively conduct experiments on five real-world datasets to demonstrate the predictive performance and scalability of our proposed method in comparison with the state-of-the-art baselines.},
  code      = {https://github.com/tund/dsgd},
  url       = {https://papers.nips.cc/paper/6560-dual-space-gradient-descent-for-online-learning},
}
```
