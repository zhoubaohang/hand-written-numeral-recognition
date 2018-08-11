# hand-written-numeral-recognition
基于神经网络的手写数字识别实验

- 一次关于神经网络应用的实验，具体介绍可以看我的博客

[博客地址](https://zhoubaohang.github.io/blog/2018/08/10/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B9%8B%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB/)

- 这次实验使用的是 **Jupyter notebook**，因此运行实验的话，直接用notebook打开 **main.ipynb**就可以了。

1. **mnist_loader.py**是用来加载**MNIST**数据集的脚本。它里面提供的函数可以直接将数据集分成 训练集、验证集、测试集。
2. **neural_network.py**是我写的一个神经网络类，里面实现了 前向、反向传播。