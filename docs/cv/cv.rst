.. knowledge_record documentation master file, created by
   sphinx-quickstart on Tue July 4 21:15:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

******************
cv
******************


一些深度学习基础知识点
============================

全连接层
-----------------
| mlp
| fully connected layers，FC
| dense 

Activation
--------------
| keras.layers.Activation(activation)
| 将激活函数应用于输出。


Dropout
---------------
| keras.layers.Dropout(rate, noise_shape=None, seed=None)
| Dropout 包括在训练中每次更新时， 将输入单元的按比率随机设置为 0， 这有助于防止过拟合。
| BN 和 dropout 的具体实现：
| BN和Dropout在训练和测试时的差别
| https://blog.csdn.net/songyunli1111/article/details/89071021

pooling池化
---------------------
| pooling池化的作用则体现在降采样：保留显著特征、降低特征维度，增大kernel的感受野。另外一点值得注意：pooling也可以提供一些旋转不变性。
| 池化层可对提取到的特征信息进行降维，一方面使特征图变小，简化网络计算复杂度并在一定程度上避免过拟合的出现；一方面进行特征压缩，提取主要特征。
| 我们的模型没做pooling

Padding
------------------
| Padding
| x = Convolution1D(8, 24, strides=2, padding='same')(x)


嵌入层 Embedding
-------------------------
| 降维。
| 它把我们的稀疏矩阵，通过一些线性变换（在CNN中用全连接层进行转换，也称为查表操作），变成了一个密集矩阵


激活函数
-------------

| 激活函数是用来加入非线性因素的，解决线性模型所不能解决的问题。
| sigmoid函数
| 前面“逻辑回归”中有介绍，非线性，输出空间在【0,1】可以直接作为输出函数，但是存在一个问题：当x很大或者很小时，函数的梯度会变得很小，利用梯度下降去收敛误差变得非常缓慢。
| Relu

.. image:: ../../_static/cv/relu.png
	:align: center
	
| tanh激活函数得到的回归线是一条曲线，而ReLU激活函数得到的是由一段一段直线构成的回归线。
| Relu 速度快  但是容易导致神经元坏死  因为一旦变成0以后梯度就永远为0了

| leakyRelu：
| 数学表达式：y = max(0, x) + leak*min(0,x)  （leak是一个很小的常数，这样保留了一些负轴的值，使得负轴的信息不会全部丢失）

.. image:: ../../_static/cv/leakyRelu.png
	:align: center




过拟合
-------------------
| 为了防止过拟合，我们需要用到一些方法，如：early stopping、数据增强（Data augmentation）、正则化（Regularization）、等。
| Early stopping方法的具体做法是，在每一个Epoch结束时（一个Epoch集为对所有的训练数据的一轮遍历）计算validation data的accuracy，当accuracy不再提高时，就停止训练。
| Dropout随机删除一些神经元防止参数过分依赖训练数据，增加参数对数据集的泛化能力


优化
-----------------------------
| SGD
| 此处的SGD指mini-batch gradient descent，关于batch gradient descent, stochastic gradient descent, 以及 mini-batch gradient descent的具体区别就不细说了。现在的SGD一般都指mini-batch gradient descent。
| SGD就是每一次迭代计算mini-batch的梯度，然后对参数进行更新，是最常见的优化方法了。
| 缺点：（正因为有这些缺点才让这么多大神发展出了后续的各种算法）
| 选择合适的learning rate比较困难 - 对所有的参数更新使用同样的learning rate。对于稀疏数据或者特征，有时我们可能想更新快一些对于不经常出现的特征，对于常出现的特征更新慢一些，这时候SGD就不太能满足要求了
| SGD容易收敛到局部最优，并且在某些情况下可能被困在鞍点

| Adam
| Adam(Adaptive Moment Estimation)本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。Adam的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。结合了adagrad和monument的优点

| •	SGD通常训练时间更长，但是在好的初始化和学习率调度方案的情况下，结果更可靠

| Momentum

| Adagrad












1x1卷积核的作用
--------------------------
| https://zhuanlan.zhihu.com/p/37910136
| 一、灵活的控制特征图的深度
| 二、减少参数 
| 三、现了跨通道的信息组合，并增加了非线性特征
| 使用1*1卷积核，实现降维和升维的操作其实就是channel间信息的线性组合变化，3*3，64channels的卷积核前面添加一个1*1，28channels的卷积核，就变成了3*3，28channels的卷积核，原来的64个channels就可以理解为跨通道线性组合变成了28channels，这就是通道间的信息交互。因为1*1卷积核，可以在保持feature map尺度不变的（即不损失分辨率）的前提下大幅增加非线性特性（利用后接的非线性激活函数），把网络做的很deep，增加非线性特性。


AUC F1 等评价指标
------------------------  
F1 score
https://www.zhihu.com/question/39840928
 
TPrate就是 预测是对的也真是对的 除以 真的是对的 TP/所有原本的T
FPrate就是 预测是对的但是是错的 除以 真的是错的 FP/所有原本的F
AUC的值即ROC曲线下的面积
AUC的优势，AUC的计算方法同时考虑了分类器对于正例和负例的分类能力，在样本不平衡的情况下，依然能够对分类器作出合理的评价
AUC的物理意义是任取一个正例和任取一个负例，正例排序在负例之前的概率。
AUC不受数据分布的影响
AUC物理意义 
精度
•	Accuracy
定义：(TP+TN)/(TP+FN+FP+TN)
即所有分类正确的样本占全部样本的比例
精确率
•	Precision、查准率
定义：(TP)/(TP+FP)
即预测是正例的结果中，确实是正例的比例
召回率
•	Recall、查全率
定义：(TP)/(TP+FN)
即所有正例的样本中，被找出的比例

F1 score
F1 = 2TP / (2TP + FP + FN)
召回率Recall和精确率Precision的几何平均数

作者：涛O_O
链接：https://www.jianshu.com/p/b425f5d9fae0
来源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


Sigmoid softmax logistics    loss
 
 


初始化
----------------
https://blog.csdn.net/xxy0118/article/details/84333635
 
