.. knowledge_record documentation master file, created by
   sphinx-quickstart on Tue July 4 21:15:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

******************
CV
******************


一些深度学习基础知识点
============================
最开始学深度学习的时候做的一些笔记，这个部分先暂时放在这里


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

**dropout 在训练和测试时候的差异**

训练时要dropout，测试和验证的时候不需要

如果失活概率为0.5，则平均每一次训练有3个神经元失活，所以输出层每个神经元只有3个输入，而实际测试时是不会有dropout的，输出层每个神经元都有6个输入，
这样在训练和测试时，输出层每个神经元的输入和的期望会有量级上的差异。

因此在训练时还要对第二层的输出数据除以（1-p）之后再传给输出层神经元，作为神经元失活的补偿，以使得在训练时和测试时每一层输入有大致相同的期望。


batch normalize BN
-------------------------------
**BN训练和测试时的参数是一样的嘛？**

| 对于BN，在训练时，是对每一批的训练数据进行归一化，也即用每一批数据的均值和方差。
| 而在测试时，比如进行一个样本的预测，就并没有batch的概念，因此，这个时候用的均值和方差是全量训练数据的均值和方差，这个可以通过移动平均法求得。
| 对于BN，当一个模型训练完成之后，它的所有参数都确定了，包括均值和方差，gamma和bata。

**BN训练时为什么不用全量训练集的均值和方差呢？**

| 因为用全量训练集的均值和方差容易过拟合，对于BN，其实就是对每一批数据进行归一化到一个相同的分布，而每一批数据的均值和方差会有一定的差别，而不是用固定的值，
这个差别实际上能够增加模型的鲁棒性，也会在一定程度上减少过拟合。
| 也正是因此，BN一般要求将训练集完全打乱，并用一个较大的batch值，否则，一个batch的数据无法较好得代表训练集的分布，会影响模型训练的效果。

**BN层具体过程、反向传播、求导**

具体过程

.. image:: ../../_static/cv/BN.png
	:align: center

求导

.. image:: ../../_static/cv/BN求导.png
	:align: center
	
	
pooling池化
---------------------
| pooling池化的作用则体现在降采样：保留显著特征、降低特征维度，增大kernel的感受野。pooling可以提供一些旋转不变性。
| 池化层可对提取到的特征信息进行降维，一方面使特征图变小，简化网络计算复杂度并在一定程度上避免过拟合的出现；一方面进行特征压缩，提取主要特征。
| 我们的模型没做pooling

| pooling的作用 
| 特征不变性，特征降维，在一定程度防止过拟合，更方便优化。


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
| sigmoid'(x)= 1/1+ e**(−x)
​
导数 s'(x) = s(x)(1-s(x))

tanh

.. image:: ../../_static/cv/tanh.png
	:align: center

| 求导
| tanh′(x)=1−tanh**2 (x)

| Relu

.. image:: ../../_static/cv/relu.png
	:align: center
	
Relu(x)=max(0,x)

relu的优点：

第一，采用sigmoid等函数，算激活函数时候（指数运算），计算量大，反向传播求误差梯度时，求导涉及除法，计算量相当大，而采用Relu激活函数，整个过程的计算量节省很多

第二，对于深层网络，sigmoid函数反向传播时，很容易就出现梯度消失的情况（在sigmoid函数接近饱和区时，变换太缓慢，导数趋于0，这种情况会造成信息丢失），从而无法完成深层网络的训练

第三，Relu会使一部分神经元的输出为0，这样就造成了网络的稀疏性，并且减少了参数的相互依存关系，缓解了过拟合问题的发生

缺点：

有一个被称为 “ReLU 死区” 的问题：在训练过程中，一些神经元会“死亡”，即它们停止输出 0 以外的任何东西。
在某些情况下，你可能会发现你网络的一半神经元已经死亡，特别是使用大学习率时。 在训练期间，如果神经元的权重得到更新，
使得神经元输入的加权和为负，则它将开始输出 0 。当这种情况发生时，由于当输入为负时，ReLU函数的梯度为0，神经元就只能输出0了。


Dead ReLU

.. image:: ../../_static/cv/Dead_ReLU.png
	:align: center


| leakyRelu：
| 数学表达式：y = max(0, x) + leak*min(0,x)  （leak是一个很小的常数，这样保留了一些负轴的值，使得负轴的信息不会全部丢失）

.. image:: ../../_static/cv/leakyRelu.png
	:align: center
	
.. image:: ../../_static/cv/softmax.png
	:align: center
	:width: 400

| tanh激活函数得到的回归线是一条曲线，而ReLU激活函数得到的是由一段一段直线构成的回归线。

 
损失函数
--------------
.. image:: ../../_static/cv/softmaxloss.png
	:align: center
	:width: 400
	
softmax loss只是交叉熵的一个特例

.. image:: ../../_static/cv/crossentropyloss.png
	:align: center
	:width: 400


.. image:: ../../_static/cv/交叉损失.png
	:align: center
	:width: 400
	
	
分类为什么用CE而不是MSE

| MSE作为分类的损失函数会有梯度消失的问题。
| MSE是非凸的，存在很多局部极小值点。

具体算一下....以前不是会推吗 ？？？好像是 用了sigmoid以后， 求一次导，链式法则，然后发现不管true label=1/-1 还是0？？导数都等于零

**有哪些损失函数**
| 深度学习中有哪些常用损失函数(优化目标函数)？ https://www.zhihu.com/question/317383780?sort=created

| Zero-one Loss（0-1损失）
| 交叉熵
| softmax loss
| KL散度
| Hinge loss
| Exponential loss与Logistic loss   (logistic loss取了Exponential loss的对数形式)

| L1 loss
| L2 loss
| smooth L1 loss (|x|<1 时等于0.5x**2,  else:等于|x|-0.5)



关于softmax细节
--------------------
更加细致的东西 

从最优化的角度看待Softmax损失函数 https://zhuanlan.zhihu.com/p/45014864

Softmax理解之二分类与多分类 https://zhuanlan.zhihu.com/p/45368976

在二分类情况下Softmax交叉熵损失等价于逻辑回归


focal loss
-------------------------
Kaiming 大神团队在他们的论文Focal Loss for Dense Object Detection 

解决分类问题中类别不平衡、分类难度差异

.. image:: ../../_static/cv/focalloss.png
	:align: center
	:width: 300

意思是这个正样本如果预测出来的概率很大，那么loss就相对小，如果预测出来概率小，那么相应的loss就大，迫使模型去更加注意那些难区分的样本
（可以自己拿个正样本，预测出来的概率是0.9试试，0.1的平方）

不难理解，α是用来适应正负样本的比例的。（如果正样本少，α为小于0.5的数，这样正样本的loss也会小）

γ称作focusing parameter，控制难易程度。

在他的模型上 α=0.25, γ=2的效果最好

为什么需要对 classification subnet 的最后一层conv设置它的偏置b为-log((1-Π)/Π)，Π代表先验概率，
就是类别不平衡中个数少的那个类别占总数的百分比，在检测中就是代表object的anchor占所有anchor的比重。论文中设置的为0.01

一开始最后一层是sigmoid，如果默认初始化情况下即w零均值，b为0，正负样本的输出都是-log(0.5)。刚开始训练的时候，loss肯定要被代表背景的anchor的误差带偏。

这样第一次，代表正样本的loss变成-log(Π), 负样本的loss变成 -log(1-Π)。正样本的loss变大

作者设置成了Π=0.01


focal loss理解与初始化偏置b设置解释 https://zhuanlan.zhihu.com/p/63626711


过拟合
-------------------
| 数据少，模型过于复杂
| 所选模型的复杂度比真模型更高;学习时选择的模型所包含的参数过多,对已经数据预测得很好,但是对未知数据预测得很差的现象.

| 为了防止过拟合，我们需要用到一些方法，如：early stopping、数据增强（Data augmentation）、正则化（Regularization）、等。
| Early stopping方法的具体做法是，在每一个Epoch结束时（一个Epoch集为对所有的训练数据的一轮遍历）计算validation data的accuracy，当accuracy不再提高时，就停止训练。
| Dropout随机删除一些神经元防止参数过分依赖训练数据，增加参数对数据集的泛化能力



如何判断过拟合还是欠拟合
------------------------------------
| 1. 欠拟合
| 一个网络是欠拟合的，那必然在开发集和验证集上的误差是很大的。假定训练集误差是20%，验证集误差是 22%，在这里对于训练集而言，误差都比较高的情况下，
说明网络对于数据集的拟合是不够的。大概率是因为网络还没训练好，应该继续训练。（高偏差）（其实就是train和val效果的数值都不好）
| •	增加特征
| •	获得更多的特征
| •	增加多项式特征
| •	减少正则化程度

| 2. 适度拟合
| 如果训练集和测试集误差都处在一个比较小，且较为相近的阶段时候，这个网络对于数据的拟合程度是比较适中的。

| 3. 过拟合
| 当继续对适度拟合的网络进行训练时候，就会造成过拟合。首先，因为对于训练集的不断学习，训练集的误差肯定会继续减小。但是于此同时训练集test loss趋于不变，
或者误差不再变化或者反而增大，训练误差和测试误差相差很大（例如训练集误差是1%，验证集误差是 18%），这个情况就要考虑是不是过拟合了。（高方差）（其实就是train和val的数值差别大）
| •	增加训练数据
| •	减少特征数量
| •	增加正则化程度
|  更多方法见上面那一条 过拟合


4. 还有一种最坏的情况，就是偏差高，方差也大。大概率就是数据集的问题了。



优化
-----------------------------
| SGD
| 此处的SGD指mini-batch gradient descent，关于batch gradient descent, stochastic gradient descent, 以及 mini-batch gradient descent的具体区别就不细说了。现在的SGD一般都指mini-batch gradient descent。
| SGD就是每一次迭代计算mini-batch的梯度，然后对参数进行更新，是最常见的优化方法了。
| 缺点：（正因为有这些缺点才让这么多大神发展出了后续的各种算法）
| 选择合适的learning rate比较困难 - 对所有的参数更新使用同样的learning rate。对于稀疏数据或者特征，有时我们可能想更新快一些对于不经常出现的特征，对于常出现的特征更新慢一些，这时候SGD就不太能满足要求了
| SGD容易收敛到局部最优，并且在某些情况下可能被困在鞍点


| •	SGD通常训练时间更长，但是在好的初始化和学习率调度方案的情况下，结果更可靠

| Momentum
.. image:: ../../_static/cv/Momentum.png
	:align: center
	
| Adagrad
.. image:: ../../_static/cv/Adagrad.png
	:align: center
	
| RMSprop
.. image:: ../../_static/cv/RMSPROP.png
	:align: center

| Adam
| Adam(Adaptive Moment Estimation)本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。Adam的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。结合了adagrad和monument的优点


.. image:: ../../_static/cv/adam.png
	:align: center

CNN参数计算
----------------------
经过CNN后输出的维度：

(N-F+2P / stride) + 1

N是上一层的image size，比如256*256。 F是filter的size，比如3*3。P是padding

参数量：
假设上一层是 227*227*3 这一层用了96个  11*11的filter
那么参数量是  3*11*11*96  注意要乘上一层的3和这一层的96


RNN LSTM Transformer的参数量见NLP那一页


1x1卷积核的作用
--------------------------
| https://zhuanlan.zhihu.com/p/37910136
| 一、灵活的控制特征图的深度
| 二、减少参数 
| 三、现了跨通道的信息组合，并增加了非线性特征
| 使用1*1卷积核，实现降维和升维的操作其实就是channel间信息的线性组合变化，3*3，64channels的卷积核前面添加一个1*1，28channels的卷积核，就变成了3*3，28channels的卷积核，原来的64个channels就可以理解为跨通道线性组合变成了28channels，这就是通道间的信息交互。因为1*1卷积核，可以在保持feature map尺度不变的（即不损失分辨率）的前提下大幅增加非线性特性（利用后接的非线性激活函数），把网络做的很deep，增加非线性特性。

.. image:: ../../_static/cv/1x1.png
	:align: center

一维卷积尺寸选取
-------------------------
主要说我们心电的项目

第一层1*15，第二层1*7，第三层1*5，后面都是1*3

理由：

| 1. 跟图片256*256不同，我们的心电数据太长了。10000个点。如果前几层不快速进行降维，后面参数压力大
| 2. 感受野。心电信号256HZ左右（降低频率之后或者不降）。一秒钟256个点，基本上15个点能大概显示有用的信息。如果一开始只有三个点，没有什么信息量。
| 3. 这样效果最好


AUC F1 等评价指标
------------------------  
| F1 score
| https://www.zhihu.com/question/39840928
 
| TPrate就是 预测是对的也真是对的 除以 真的是对的 TP/所有原本的T
| FPrate就是 预测是对的但是是错的 除以 真的是错的 FP/所有原本的F

.. image:: ../../_static/cv/TPrate.png
	:align: center


| AUC的值即ROC曲线下的面积
| AUC的优势，AUC的计算方法同时考虑了分类器对于正例和负例的分类能力，在样本不平衡的情况下，依然能够对分类器作出合理的评价
| AUC的物理意义是任取一个正例和任取一个负例，正例排序在负例之前的概率。
| AUC不受数据分布的影响
| AUC物理意义 

.. image:: ../../_static/cv/AUC.png
	:align: center

| 精度
| •	Accuracy
| 定义：(TP+TN)/(TP+FN+FP+TN)
| 即所有分类正确的样本占全部样本的比例
| 精确率
| •	Precision、查准率
| 定义：(TP)/(TP+FP)
| 即预测是正例的结果中，确实是正例的比例
| 召回率
| •	Recall、查全率
| 定义：(TP)/(TP+FN)
| 即所有正例的样本中，被找出的比例

| F1 score
| F1 = 2TP / (2TP + FP + FN)
| 召回率Recall和精确率Precision的几何平均数

| 链接：https://www.jianshu.com/p/b425f5d9fae0


top1 error， top5 error

| top1 error（正确标记 与 模型输出的最佳标记不同的样本数）/ 总样本数
| 只能猜一次，猜错的概率

| top5  error（正确标记 不在 模型输出的前5个最佳标记中的样本数）/ 总样本数
| 能猜五个，五个都猜不中的概率


初始化
----------------
https://blog.csdn.net/xxy0118/article/details/84333635
 
.. image:: ../../_static/cv/初始化.png
	:align: center



反向传播的推导
------------------------
以前做过的作业  gradescope

.. image:: ../../_static/cv/homework1.png
	:align: center
	
.. image:: ../../_static/cv/homework2.png
	:align: center

池化层如何反向传播 
-------------------------
https://blog.csdn.net/weixin_41683218/article/details/86473488

.. image:: ../../_static/cv/mean_pooling.png
	:align: center
	
.. image:: ../../_static/cv/max_pooling.png
	:align: center
	
loss下降不下降的问题
----------------------------
https://blog.csdn.net/zongza/article/details/89185852


梯度爆炸 梯度消失
-------------------------------
反向传播时，如果网络过深，每层梯度连乘小于1的数，值会趋向0，发生梯度消失。大于1则趋向正无穷，发生梯度爆炸。

梯度爆炸 — 梯度剪裁 ：如果梯度过大则投影到一个较小的尺度上

梯度消失 — 使用ReLU, Batch Norm，Xavier初始化和He初始化



CV的一些知识
===================

各类模型
------------
AlexNet  VGG  GoogleNet  ResNet  DenseNet

马上上线

HighwayNetworks
---------------------------------
Highway Network保留了ResNet中的短路通道，但是可以通过可学习的参数来加强它们，以确定在哪层可以跳过，哪层需要非线性连接。

其实所谓Highway网络，无非就是输入某一层网络的数据一部分经过非线性变换，另一部分直接从该网络跨过去不做任何转换，就像走在高速公路上一样，
而多少的数据需要非线性变换，多少的数据可以直接跨过去，是由一个权值矩阵和输入数据共同决定的。