.. knowledge_record documentation master file, created by
   sphinx-quickstart on Tue July 4 21:15:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

******************
NLP
******************

GRU LSTM BRNN
=====================
吴恩达https://www.bilibili.com/video/BV1F4411y7BA?p=9

.. image:: ../../_static/nlp/GRULSTM.png
	:align: center

.. image:: ../../_static/nlp/lstm.png
	:align: center

RNN的弊端，还有LSTM内部结构，以及接收的是前一个LSTM的什么？怎样解决长期依赖？为什么要用sigmoid?

长期依赖，三个门，加计算公式，sigmoid将值限制在了0-100%


 

bert & transformer
=================

马上上线

一些学习资料
----------------------
李宏毅 transformer讲解视频：
https://www.bilibili.com/video/BV1J441137V6?from=search&seid=1952161104243826844

https://blog.csdn.net/urbanears/article/details/98742013  这个博客讲的不错



-----------------

知识蒸馏

剪枝

| crf
| n gram
| attention
| transformer
| gpt
| bert
| bagofword
| fasttext
| glove
| elmo
| 知识图谱
| 模型压缩的相关知识三大角度：蒸馏，剪枝，量化

意图识别
编辑距离
elasticsearch
召回再匹配
fastbert

知识图谱

https://cloud.tencent.com/developer/article/1558479


.. image:: ../../_static/nlp/transformer.png
	:align: center



https://zhuanlan.zhihu.com/p/148656446

史上最全Transformer面试题

Transformer为何使用多头注意力机制？（为什么不使用一个头）
---------------------------------------------------------------

这个目前还没有公认的解释，本质上是论文原作者发现这样效果确实好。但是普遍的说法是，使用多个头可以提供多个角度的信息。

在同一“multi-head attention”层中，输入均为“KQV”，同时进行注意力的计算，彼此之前参数不共享，最终将结果拼接起来，这样可以允许模型在不同的表示子空间里学习到相关的信息

Transformer为什么Q和K使用不同的权重矩阵生成，为何不能使用同一个值进行自身的点乘？
-----------------------------------------------------------------------------------------

.. image:: ../../_static/nlp/self-attention.png
	:align: center
	:width: 400

| 简单解释：
| Q和K的点乘是为了计算一个句子中每个token相对于句子中其他token的相似度，这个相似度可以理解为attetnion score

| 复杂解释：
| 经过上面的解释，我们知道K和Q的点乘是为了得到一个attention score 矩阵，用来对V进行提纯。
| K和Q使用了不同的W_k, W_Q来计算，可以理解为是在不同空间上的投影。
| 正因为有了这种不同空间的投影，增加了表达能力，这样计算得到的attention score矩阵的泛化能力更高。
| 这里解释下我理解的泛化能力，因为K和Q使用了不同的W_k, W_Q来计算，得到的也是两个完全不同的矩阵，所以表达能力更强。

链接：https://www.zhihu.com/question/319339652/answer/730848834

Transformer计算attention的时候为何选择点乘而不是加法？两者计算复杂度和效果上有什么区别？
-----------------------------------------------------------------------------------------------------------
| 为了计算更快。
| 矩阵加法在加法这一块的计算量确实简单，但是作为一个整体计算attention的时候相当于一个隐层，整体计算量和点积相似。
| 在效果上来说，从实验分析，两者的效果和dk相关，dk越大，加法的效果越显著。

为什么在进行softmax之前需要对attention进行scaled（为什么除以dk的平方根），并使用公式推导进行讲解
-------------------------------------------------------------------------------------------------------------
作者在分析模型性能不佳的原因时，认为是极大的点积值将落在 softmax 平缓区间，使得收敛困难。类似“梯度消失”。

（洛）为什么其他的softmax不用scaled？因为以前类似于softmaxloss的时候，计算的是logits的softmax，是模型预测结果和真实值的差异，本来就不会出现一个很大的值。



在计算attention score的时候如何对padding做mask操作？
-------------------------------------------------------------------
padding位置置为负无穷(一般来说-1000就可以)

为什么在进行多头注意力的时候需要对每个head进行降维？（可以参考上面一个问题）
--------------------------------------------------------------------------------------------
额。。。这里顺便把QKV也讲一下

.. image:: ../../_static/nlp/QKV.png
	:align: center
	:width: 500

.. image:: ../../_static/nlp/QKVSMALL.png
	:align: center
	:width: 150
	
一个 max_seq_len=210的句子，如上图中的x，通过word embedding，得到一个210*256的矩阵，如上图a。

a 通过线性层（但还是210*256------210*256）,得到 QKV 三个矩阵，其实基本可以看成a'。然后通过上面的公式 Q和K(T)做点乘，除以dimensionK，然后softmax，然后乘V

上面通过的这个线性层，我们的实际经验是，去掉线性层这个步骤效果没区别，而且参数量小很多。

当多头的时候，比如8头，处理方法是210*256的矩阵变成  8个 210*32 的矩阵，然后multi-head attention的计算完成后再拼接起来。

作者的目的应该是怕头多了以后，在高维空间中参数量大，学习效果不好。

https://www.cnblogs.com/rosyYY/p/10115424.html
这篇文章讲的multihead不错，截图在下方：

.. image:: ../../_static/nlp/multihead.png
	:align: center

Transformer的Encoder模块
-----------------------------------

| 8. 为何在获取输入词向量之后需要对矩阵乘以embedding size的开方？意义是什么？
| 9. 简单介绍一下Transformer的位置编码？有什么意义和优缺点？

.. image:: ../../_static/nlp/positional_encoding.png
	:align: center

.. image:: ../../_static/nlp/wp.png
	:align: center


| 11. 简单讲一下Transformer中的残差结构以及意义。



简单描述一下Transformer中的前馈神经网络？使用了什么激活函数？相关优缺点？
--------------------------------------------------------------------------------------------
Feed Forward层是一个两层的fully-connection层，中间有一个ReLU激活函数，隐藏层的单元个数为 2048。

是非线性变换，能增强学习能力

Encoder端和Decoder端是如何进行交互的？（在这里可以问一下关于seq2seq的attention知识）
--------------------------------------------------------------------------------------------------
Q矩阵来源于下面子模块的输出（对应到图中即为masked多头self-attention模块经过Add&Norm后的输出），而K，V矩阵则来源于整个Encoder端的输出

Decoder阶段的多头自注意力和encoder的多头自注意力有什么区别？（为什么需要decoder自注意力需要进行 sequence mask)
---------------------------------------------------------------------------------------------------------------------------
Decoder端的多头self-attention需要做mask，因为它在预测时，是“看不到未来的序列的”，所以要将当前预测的单词（token）及其之后的单词（token）全部mask掉。

| 17. Transformer的并行化提现在哪个地方？Decoder端可以做并行化吗？

Transformer训练的时候学习率是如何设定的？Dropout是如何设定的，位置在哪里？Dropout 在测试的需要有什么需要注意的吗？
-------------------------------------------------------------------------------------------------------------------------------
| bert里面是1e-5，我自己尝试过1e-6和5e-6，效果略微下降
| dropout的话，代码里是0.1 但这是训练的时候。我测试的时候最开始忘记dropout设为零了，后来发现了以后直接有1个百分点的提升。

具体细节？？？？？？？位置在哪里？




self-attention的优点
----------------------------------
引入self-attention后会更容易捕获句子中长距离的相互依赖特征。
因为如果是LSTM或者RNN，需要依次序序列计算，对于远距离的相互依赖的特征，要经过若干时间步步骤的信息累积才能将两者联系起来，而距离越远，有效捕获的可能性越小。

self-attention在计算过程中会直接将句子中任意两个单词的联系通过一个计算步骤直接联系起来，
所以远距离依赖特征之间的距离被极大缩短，有利于有效地利用这些特征。

除此外，self-attention对计算的并行性也有直接帮助。

关于并行计算
--------------------
Encoder端可以并行计算，一次性将输入序列全部encoding出来，但Decoder端不是一次性把所有单词（token）预测出来的，而是像seq2seq一样一个接着一个预测出来的

前馈网络和BP神经网络
--------------
前馈神经网络主要强调的是无环

BP网络指的是用BP算法进行训练的多层前馈神经网络

batch norm & layer norm
-----------------------------------
这个说的勉强还行：https://zhuanlan.zhihu.com/p/74516930

| '''
| 而LN则是针对一句话进行缩放的，且LN一般用在第三维度，如[batchsize, seq_len, dims]中的dims，一般为词向量的维度，或者是RNN的输出维度等等，
| 这一维度各个特征的量纲应该相同。因此也不会遇到上面因为特征的量纲不同而导致的缩放问题。
| '''

batch normalization 受batch size的影响很大，因为他是用batch里的数据来假设是整体样本的情况

https://zhuanlan.zhihu.com/p/54530247

为什么BN不太适用于RNN：

.. image:: ../../_static/nlp/BNRNN.png
	:align: center

由于文本的长度不一致，例如这几句话，当句长大于4的时候，就只有一个样本了（剩下的全是padding）。做batch norm的话size太小，不能反映样本的整体分布。


warmup
-------------