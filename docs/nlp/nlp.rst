.. knowledge_record documentation master file, created by
   sphinx-quickstart on Tue July 4 21:15:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

******************
NLP
******************

项目涉及
=====================

（NLP）语义分析--文本分类、情感分析、意图识别
--------------------------------------------------------
https://blog.csdn.net/weixin_41657760/article/details/93163519

摘录一下意图识别部分

.. image:: ../../_static/nlp/意图识别.png
	:align: center



模型压缩
------------------
模型压缩的相关知识三大角度：蒸馏，剪枝，量化

.. image:: ../../_static/nlp/模型压缩.png
	:align: center
	:width: 400
	
.. image:: ../../_static/nlp/模型压缩2.png
	:align: center
	:width: 400
	
	
知识蒸馏
------------------
李宏毅是真的讲得好https://www.bilibili.com/video/BV1SC4y1h7HB?p=7

.. image:: ../../_static/nlp/distillation.png
	:align: center
	:width: 500

| 为什么distillation有效果。因为teacher的参数能提供比label data（one-hot label）更多的信息。
| 比如上面那张图，如果student从teacher那里学习，他不仅能知道这张图片是1，而且知道1和7有点像，也知道1和9有点像

.. image:: ../../_static/nlp/temperature.png
	:align: center
	:width: 400

temperature就是为了防止teacher的反馈和one-hot label太像



编辑距离
-----------------
一个单词转换为另一个单词所需要的最少单字符编辑操作次数

具体计算过程 https://zhuanlan.zhihu.com/p/91667128

？？ 再看，代码要会写


fastbert
-----------------
ACL2020一篇关于提高BERT推理速度的文章，提出了一种新的inference速度提升方式，相比单纯的student蒸馏有更高的确定性，且可以自行权衡效果与速度

FastBERT的创新点很容易理解，就是在每层Transformer后都加分类器去预测样本标签，如果某样本预测结果的置信度很高，就不用继续计算了。

论文把这个逻辑称为样本自适应机制（Sample-wise adaptive mechanism），就是自适应调整每个样本的计算量，容易的样本通过一两层就可以预测出来，较难的样本则需要走完全程。

这里的分支Classifier都是最后一层的分类器蒸馏来的，作者将这称为自蒸馏（Self-distillation）。

就是在预训练和精调阶段都只更新主干参数，精调完后freeze主干参数，用分支分类器（图中的student）蒸馏主干分类器（图中的teacher）的概率分布。

之所以叫自蒸馏，是因为之前的蒸馏都是用两个模型去做，一个模型学习另一个模型的知识，而FastBERT是自己（分支）蒸馏自己（主干）的知识。

.. image:: ../../_static/nlp/fastbert_uncertainty.png
	:align: center
	:width: 400
	
不确定性就是用熵来衡量的。熵越大代表结果越不可信，如果某一层的不确定性小于一个阈值，那么我们就对这层的结果进行输出，从而提高了推理速度


elasticsearch（es）/倒排索引
------------------------------------
简单的说法

.. image:: ../../_static/nlp/倒排索引.png
	:align: center

https://blog.csdn.net/RuiKe1400360107/article/details/103864216

更详细的再去网上搜



召回
------------
协同过滤，聚类

搜索引擎的两大问题（1） - 召回https://www.douban.com/note/722330114/

| 索引粒度问题。
我们知道召回是通过倒排索引求交得到的，当以词为粒度，粒度较细，
召回的文章的数目较多，但也可能由于倒排过长把一些相关的结果误截断；当以更大的phrase粒度，
粒度较粗，召回的文章相对更相关，但也容易造成召回的结果过少。

| 我们的项目里面为什么那样召回：
| 1.数据量和复杂程度只有这么高
| 2.业务那边对时间有较强的要求
| 3.的确这样的效果很好，给业务和上级汇报的时候可解释性也很强 


匹配
--------------
| 排序学习 Learning to Rank
| https://lumingdong.cn/learning-to-rank-in-recommendation-system.html

pointwise、pairwise、listwise
------------------------------------------
https://zhuanlan.zhihu.com/p/56938216

在pointwise中把排序问题当成一个二分类问题，训练的样本被组织成为一个三元组（qi，cij，yij）跟我们的构造方法一样

在pairwise方法中排序模型让正确的回答的得分明显高于错误的候选回答。给一个提问，pairwise给定一对候选回答学习并预测哪一个句子才是提问的最佳回答。

训练的样例为（qi，ci+，ci-）,其中qi为提问,ci+为正确的回答，ci-为候选答案中一个错误的回答。

损失函数为合页损失函数

listwise：  pariwise和pointwise忽视了一个事实就是答案选择就是从一系列候选句子中的预测问题。在listwise中单一训练样本就是提问数据和它的所有候选回答句子。
在训练过程中给定提问和它的一系列候选句子和标签

ernie
------------
ERNIE 沿袭了 BERT 中绝大多数的设计思路，包括 预训练 (Pretraining) 加 微调 (Fine-tuning) 的流程，
去噪自编码 (DAE, abbr. denoising autoencoding) 的模型本质，以及 Masked Language Model 和 
Next Sentence Prediction 的训练环节。主要的不同，在于 ERNIE 采用了更为复杂的 Masking 策略：
Knowledge Masking Strategies，并针对对话型数据引入一套新的训练机制：对话语言模型 (Dialogue Language Model)。


ERNIE2是百度在ERNIE1基础上的一个升级版，不过这次升级幅度比较大. ERNIE 2.0 将 1.0 版本中的功能特性全部予以保留，
并在此基础上做更为丰富的扩展和延伸。论文指出，近几年来基于未标注语料进行无监督编码的预训练模型，
包括 Word2Vec、ELMo、GPT、BERT、XLNet、ERNIE 1.0， 存在一个共同缺陷：仅仅只是利用了token与token之间的共现(Co-occurance) 信息。
当两个 token 拥有相似的上下文语境时，最终的编码必然具有极高的相似度。这使得模型无法在词向量中嵌入语料的 词汇 (lexical)、语法 (syntatic) 
以及 语义 (semantic) 信息。为此，ERNIE 2.0 首次引入 连续预训练 (Continual Pre-training) 机制 —— 以串行的方式进行多任务学习，学习以上三类特征。
设计的初衷在于模拟人类的学习行为：利用已经积累的知识，持续地进行新的学习。


albert
----------------
ALBERT的贡献

文章里提出一个有趣的现象：当我们让一个模型的参数变多的时候，一开始模型效果是提高的趋势，但一旦复杂到了一定的程度，接着再去增加参数反而会让效果降低，这个现象叫作“model degratation"。

基于上面所讲到的目的，ALBERT提出了三种优化策略，做到了比BERT模型小很多的模型，但效果反而超越了BERT， XLNet。

- Factorized Embedding Parameterization. 他们做的第一个改进是针对于Vocabulary Embedding。在BERT、XLNet中，
词表的embedding size(E)和transformer层的hidden size(H)是等同的，所以E=H。但实际上词库的大小一般都很大，
这就导致模型参数个数就会变得很大。为了解决这些问题他们提出了一个基于factorization的方法。

他们没有直接把one-hot映射到hidden layer, 而是先把one-hot映射到低维空间之后，再映射到hidden layer。这其实类似于做了矩阵的分解。

- Cross-layer parameter sharing. Zhenzhong博士提出每一层的layer可以共享参数，这样一来参数的个数不会以层数的增加而增加。所以最后得出来的模型相比BERT-large小18倍以上。

- Inter-sentence coherence loss. 在BERT的训练中提出了next sentence prediction loss, 也就是给定两个sentence segments, 然后让BERT去预测它俩之间的先后顺序，但在ALBERT文章里提出这种是有问题的，其实也说明这种训练方式用处不是很大。 
所以他们做出了改进，他们使用的是setence-order prediction loss (SOP)，其实是基于主题的关联去预测是否两个句子调换了顺序。

只用了四层的transformer，但是效果下降不多。



RoBERTa
-------------------
| 从模型上来说，RoBERTa基本没有什么太大创新，主要是在BERT基础上做了几点调整：
| 1）动态Masking，相比于静态，动态Masking是每次输入到序列的Masking都不一样；
| 2）移除next predict loss，相比于BERT，采用了连续的full-sentences和doc-sentences作为输入（长度最多为512）；
| 3）更大batch size，batch size更大，training step减少，实验效果相当或者更好些；
| 4）text encoding，基于bytes的编码可以有效防止unknown问题。另外，预训练数据集从16G增加到了160G，训练轮数比BERT有所增加。

静态Masking vs 动态Masking

原来Bert对每一个序列随机选择15%的Tokens替换成[MASK]，为了消除与下游任务的不匹配，还对这15%的Tokens进行
（1）80%的时间替换成[MASK]；（2）10%的时间不变；（3）10%的时间替换成其他词。
但整个训练过程，这15%的Tokens一旦被选择就不再改变，也就是说从一开始随机选择了这15%的Tokens，之后的N个epoch里都不再改变了。这就叫做静态Masking。

而RoBERTa一开始把预训练的数据复制10份，每一份都随机选择15%的Tokens进行Masking，也就是说，
同样的一句话有10种不同的mask方式。然后每份数据都训练N/10个epoch。这就相当于在这N个epoch的训练中，每个序列的被mask的tokens是会变化的。这就叫做动态Masking。




解释我们的NSP：next sentence prediction
-----------------------------------------------------
| 为什么大家都说nsp效果不好，我觉得应该是数据太简单了。就是他的后文选取的太随意。

NSP是预训练，我们这个是下游任务

但是我们的不是，我们的数据很难，比如第一句是汉武大帝这部电影，我们第二句的数据在构造的时候会包括汉武大帝这部电影，
会包括汉武大帝这个人，会包括汉武大帝这本书，汉武大帝的纪录片等等，模型一定要深入理解了内在关系才能进行判断

可以扯一下albert的sop



基础知识
==================

剪枝

| crf

word2vec
--------------------
李宏毅机器学习 unsupervised learning word embedding  https://www.bilibili.com/video/BV13x411v7US?p=25

这个是别人做的笔记 https://www.jianshu.com/p/8c7030ccbf9e

核心：通过一个词汇的上下文去了解他的意思

count base的做法

wi词和wj词查找共同出现的次数，那么wi词的向量和wj词的向量的乘积应该接近于这个数

.. image:: ../../_static/nlp/count_base.png
	:align: center
	:width: 400

predition base的做法

.. image:: ../../_static/nlp/predition_base.png
	:align: center
	:width: 400

用前一个字去预测后一个字。先用one-hot编码，然后输入到一个神经网络，接着去预测。那么他把神经网络的第一层（相当于是一个降维的功能）当作word embedding

如果不仅是只用前一个词，而是前N个词。那么w这个权重是共享的。

.. image:: ../../_static/nlp/predition_base2.png
	:align: center
	:width: 400

变体：

.. image:: ../../_static/nlp/变体.png
	:align: center
	:width: 400


CBOW 用两边预测中间。 skip-gram 用中间预测两边

skip-gram 训练时间更长，出来的准确率比cbow 高。但是生僻词较多时cbow更好

优化：

hierarchical softmax优化、负采样

Negative Sampling是对于给定的词,并生成其负采样词集合的一种策略,已知有一个词,这个词可以看做一个正例,而它的上下文词集可以看做是负例,
但是负例的样本太多,而在语料库中,各个词出现的频率是不一样的,所以在采样时可以要求高频词选中的概率较大,
低频词选中的概率较小,这样就转化为一个带权采样问题,大幅度提高了模型的性能。

hierarchical softmax 将词库表示成前缀树，从树根到叶子的路径可以表示为一系列二分类器，一次多分类计算的复杂度从|V|降低到了树的高度

Word2vec ------算法岗面试题
https://www.cnblogs.com/zhangyang520/p/10969975.html





bag_of_word
--------------------
它的基本思想是假定对于一个文本，忽略其词序和语法、句法，仅仅将其看做是一些词汇的集合，而文本中的每个词汇都是独立的。
简单说就是将每篇文档都看成一个袋子（因为里面装的都是词汇，所以称为词袋，Bag of words即因此而来），然后根据袋子里装的词汇对其进行分类。
如果文档中猪、马、牛、羊、山谷、土地、拖拉机这样的词汇多些，而银行、大厦、汽车、公园这样的词汇少些，我们就倾向于判断它是一篇描绘乡村的文档，而不是描述城镇的。

各种词语出现的数量

n-gram
-------------------

.. image:: ../../_static/nlp/n-gram.png
	:align: center


fasttext
------------------------
字符级n-gram特征的引入以及分层Softmax分类。


glove
----------------
word2vec是“predictive”的模型，而GloVe是“count-based”的模型


.. image:: ../../_static/nlp/ELMO.png
	:align: center

先进来w1，然后得到隐藏层h1，通过线性层和softmax预测下一句。所以w1 w2 w3不能一次性全读入

ELMO双向： 

.. image:: ../../_static/nlp/ELMO双向.png
	:align: center
	:width: 400
	
从前往后有个LSTM生成隐层，从后往前也有个LSTM生成隐层，然后两个隐层拼接起来才是总共的embedding结果

但是相比BERT存在的问题是：从前往后的时候只看到这个词为止了，没有继续往后看，从后往前的时候也是没有看到最开始。
	
.. image:: ../../_static/nlp/bertmask.png
	:align: center
	:width: 400

bert的话，把w2遮住或者随机替换。用隐藏层去预测w2。这样的话，隐藏层里面会看见前后的所有信息。


细粒度分类
----------------------
？？？？待补充


XLNet
------------
transformer-XL  具体细节待补充？？？

nlp中的数据增强
----------------------
随机drop和shuffle、同义词替换、回译、文档裁剪、GAN、预训练的语言模型


RNN及其变体LSTM等
=============================================
这个从基础知识里面单独拿出来讲

完全图解RNN、RNN变体、Seq2Seq、Attention机制 https://www.leiphone.com/news/201709/8tDpwklrKubaecTa.html

简单版RNN

.. image:: ../../_static/nlp/RNN.png
	:align: center
	:width: 200

例如在生成x2的隐藏层h2的时候，h2 = f(Ux2 + Wh1 + b)。然后依次计算剩下来的 **（使用相同的参数U、W、b）**

生成输出y1的时候，y1 = Softmax(Vh1 + c) 。剩下的输出类似进行 **（使用和y1同样的参数V和c）**

为了输入输出不等长，所以出现了seq2seq （encoder decoder）

.. image:: ../../_static/nlp/seq2seq.png
	:align: center
	:width: 300


GRU LSTM BRNN
--------------------------------
吴恩达https://www.bilibili.com/video/BV1F4411y7BA?p=9

.. image:: ../../_static/nlp/GRULSTM.png
	:align: center

.. image:: ../../_static/nlp/lstm.png
	:align: center

RNN的弊端，还有LSTM内部结构，以及接收的是前一个LSTM的什么？怎样解决长期依赖？为什么要用sigmoid?

长期依赖，三个门，加计算公式，sigmoid将值限制在了0-100%

.. image:: ../../_static/nlp/LSTM复杂度.png
	:align: center




attention
===================
不错的资料
-------------------
自然语言处理中的Attention机制总结 https://blog.csdn.net/hahajinbu/article/details/81940355


直观的解释
---------------------
| 核心就是权重。 比如汤姆追逐杰瑞，tom chase jerry。那么生成tom的时候，肯定是汤姆生成的隐藏层会占比巨大。那么如何得到权重呢，就算汤姆生成的隐藏层和Tom的隐藏层去做点积。
| 在余弦相似度里面我们知道，如果两个向量相似，那么他们的cos会接近1。所以这样分分别计算，再softmax，就是权重。（其实具体过程和self-attention基本一致）

为什么要引入Attention机制
-----------------------------------
计算能力的限制：当要记住很多“信息“，模型就要变得更复杂，然而目前计算能力依然是限制神经网络发展的瓶颈。

优化算法的限制：虽然局部连接、权重共享以及pooling等优化操作可以让神经网络变得简单一些，有效缓解模型复杂度和表达能力之间的矛盾；
但是，如循环神经网络中的长距离以来问题，信息“记忆”能力并不高。


手写
--------------
？？？待补充

attention的一个通用定义
----------------------------------
按照Stanford大学课件上的描述，attention的通用定义如下：

| 给定一组向量集合values，以及一个向量query，attention机制是一种根据该query计算values的加权求和的机制。
| attention的重点就是这个集合values中的每个value的“权值”的计算方法。
| 有时候也把这种attention的机制叫做query的输出关注了（或者说叫考虑到了）原文的不同部分。（Query attends to the values）
| 举例：刚才seq2seq中，哪个是query，哪个是values？
| each decoder hidden state attends to the encoder hidden states （decoder的第t步的hidden state----st是query，encoder的hidden states是values）


变体
-------------
答案是我自己根据一些博客总结的，待进一步考究与确认

| Soft attention
| 就是上面说的那种最普通的attention
| 照顾到全部的位置，只是不同位置的权重不同

| Hard attention
| 取最大的地方为1，其他的为0。存在的问题是不可导，要用蒙特卡洛方法对 s 进行抽样
| 直接从输入句子里面找到某个特定的单词，然后把目标句子单词和这个单词对齐，而其它输入句子中的单词硬性地认为对齐概率为0


| local attention（“半软半硬”的attention）
| 使用了一个人工经验设定的参数D去选择一个以pt为中心，[pt−D,pt+D]为窗口的区域，进行对应向量的weighted sum


动态attention

Attention score的计算方式变体

.. image:: ../../_static/nlp/attention_score.png
	:align: center
	:width: 400

静态attention

强制前向attention

self attention

key-value attention

multi-head attention


bert
====================

一些学习资料
----------------------

Bert的一些面试题 https://cloud.tencent.com/developer/article/1558479
 
这篇文章讲解bert还不错 https://zhuanlan.zhihu.com/p/46652512

李宏毅bert https://www.bilibili.com/video/BV1C54y1X7xJ?p=1

BERT 时代的常见 NLP 面试题 https://blog.csdn.net/qq_34832393/article/details/104356462


embedding
--------------------

.. image:: ../../_static/nlp/embedding.png
	:align: center
	:width: 400

| • Token Embeddings是词向量，第一个单词是CLS标志，可以用于之后的分类任务
| • Segment Embeddings用来区别两种句子，因为预训练不光做LM还要做以两个句子为输入的分类任务
| • Position Embeddings和之前文章中的Transformer不一样，不是三角函数而是学习出来的


GPT 与 BERT 的区别是什么
-------------------------------
？？？ 待补充

GPT 是单向的，BERT 是双向的。

训练方法不同，GPT 是语言模型使用最大似然，BERT 的训练分为 MLM 和 NSP。

GPT Bert与XLNet的差异
https://cloud.tencent.com/developer/article/1507551

.. image:: ../../_static/nlp/GPTBERTXLNET.png
	:align: center
	
	
BERT 与BiLSTM 有什么不同
-----------------------------------


transformer
=================

一些学习资料
----------------------
李宏毅 transformer讲解视频：
https://www.bilibili.com/video/BV1J441137V6?from=search&seid=1952161104243826844

Transformer模型中重点结构详解 https://blog.csdn.net/urbanears/article/details/98742013  这个博客讲的不错

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

记得多头之间参数不共享

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

具体细节？？？？？？？位置在哪里？待补充




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
	:width: 300

由于文本的长度不一致，例如这几句话，当句长大于4的时候，就只有一个样本了（剩下的全是padding）。做batch norm的话size太小，不能反映样本的整体分布。


warmup
-------------


CRF 条件随机场
==========================

一些资料
----------------
条件随机场（CRF） 举例讲解 https://www.cnblogs.com/sss-justdDoIt/p/10218886.html

CRF系列(一)——一个简单的例子 https://zhuanlan.zhihu.com/p/69849877

sequence labeling problem 李宏毅 https://www.bilibili.com/video/BV1zJ411575b?from=search&seid=2817357212456451101

通俗易懂理解——BiLSTM-CRF https://zhuanlan.zhihu.com/p/115053401


生成式判别式
--------------------
假定所关心的变量集合为Y, 可观测变量集合为O，其他变量的集合为R，

"生成式" (generative) 模型考虑联合分布 P(Y,O,R) 

判别式" (discriminative)型考虑条件分布 P(Y,R|O). 

给定一组观测变量值，推断就是要由P(Y,O,R)或P(Y,R|O)得到条件概率分布 P(Y|O).


HMM
-----------------
| HMM模型五要素<L,W,A,B,Π>
| L是状态集，一共有多少种词性
| W是词典
| A是状态转移矩阵，可以理解成语法规则。前面一个是副词，后面一个是副词的概率也很大。根据语料计算出来的
| B是在不同的词性的情况下，词的概率
| Π是文本第一个词的词性分布

| aij = Aij/sum 1<=j<=N (Aij)  aij是求分布。Aij是计数，前面一个词性是i，后面一个词性是j的数量。
| Bjk也是一样，词性j下，词语为k的计数

.. image:: ../../_static/nlp/HMM1.png
	:align: center
	:width: 400
	
.. image:: ../../_static/nlp/HMM2.png
	:align: center
	:width: 400
	
.. image:: ../../_static/nlp/HMM3.png
	:align: center
	:width: 400


CRF
---------------------------------
| CRF 认为 p(x,y)正比于exp(w*Φ(x,y)) 。 其中，w是权重，Φ(x,y)是特征向量。然后exp(w*Φ(x,y))永远是正数，而且有时会大于1，所以不是概率但像概率
| 两边取log，有log(p(x,y))正比于w*Φ(x,y)

.. image:: ../../_static/nlp/CRF1.png
	:align: center
	:width: 400

| 下面那个式子是针对红框里的
| s是tags，词性，    t是具体的词
| p(t|s)是给定词性，词语是t的概率
| Ns,t(x,y)是s和t共现的数量

.. image:: ../../_static/nlp/CRF2.png
	:align: center
	:width: 400
	
w是训练中要学习的权重

.. image:: ../../_static/nlp/CRF3.png
	:align: center
	:width: 400

所以，训练的时候要寻找一个权重w，做到O(w)表达的那个式子最大

拆开写，由两部分组成。已出现过的共现对要权重大，未出现过的搭配权重要小。

.. image:: ../../_static/nlp/CRF4.png
	:align: center
	:width: 400
	
反向传播 求导


Bi-LSTM，CRF
-------------------------

| 发射矩阵  lstm
| 转移矩阵 crf

.. image:: ../../_static/nlp/bilstm-crf.png
	:align: center
	:width: 400

从上图可以看出，BiLSTM层的输出是每个标签的得分，如单词w0，BiLSTM的输出为1.5（B-Person），0.9（I-Person），0.1(B-Organization), 0.08 (I-Organization) and 0.05 (O)，
这些得分就是CRF层的输入。

将BiLSTM层预测的得分喂进CRF层，具有最高得分的标签序列将是模型预测的最好结果。

如果没有CRF层将如何：无视语法规则，比如动词后面还会继续接动词等

在CRF层的损失函数中，有两种类型的得分，这两种类型的得分是CRF层的关键概念。

第一个得分为发射得分，该得分可以从BiLSTM层获得。第二个是转移得分，这是BiLSTM-CRF模型的一个参数，在训练模型之前，
可以随机初始化该转移得分矩阵，在训练过程中，这个矩阵中的所有随机得分将得到更新，换而言之，
CRF层可以自己学习这些约束条件，而无需人为构建该矩阵。随着不断的训练，这些得分会越来越合理。


维特比算法
------------------------
最短路径和？？？再看

知识图谱
====================


实体识别 关系识别 实体消歧 还有啥是知识图谱 
？？？？？？？？？待补充

概念解释
-------------------------
【NLP笔记】知识图谱相关技术概述  https://zhuanlan.zhihu.com/p/153392625

| 「实体识别」
大家更常指的是NER命名实体识别，指识别文本中具有特定意义的实体，主要包括人名、地名、机构名、专有名词等，以及时间、数量、货币、比例数值等文字。
一般用序列标注问题来处理实体识别。传统的实体识别方法以统计模型如HMM、CRF等为主导，随着深度学习的兴起，
DNN+CRF的结构非常的通用，DNN部分可以使用一些主流的特征抽取器如Bi-LSTM, Bert等，这种模型在数据不至于太糟的情况下，轻易就能有90%+的效果。

| 「实体对齐」
具有不同标识实体（ID标识符）却代表真实世界中同一对象的那些实体，
并将这些实体归并为一个具有全局唯一标识的实体对象添加到知识图谱中，即同一个实质不同的名字，需要将这些本质相同的东西归并。

| 「实体消岐」
是用于解决同个实体名称在不同语句不同意义的问题，同一个词不同的实质，
如apple 在知识图谱里至少有两个歧义，静态词向量无法解决这个问题，不能要求你的知识库里所有的苹果 ，都是吃的苹果。

| 「实体链接」
将自由文本中已识别的实体对象（人名、地名、机构名等），无歧义的正确的指向知识库中目标实体的过程。
本质仍是同一个词不同实质，如已有一个知识库的情况下，预测输入query的某个实体对应知识库id，
如 apple 在一个有上下文的query中是指能吃的apple 还是我们手上用的apple。实体链接强调链接的过程，而消岐强调先描述这个实体是什么。


知识图谱的存储

一种是基于 RDF 的存储；另一种是基于图数据库的存储（使用更多）。
RDF 一个重要的设计原则是数据的易发布以及共享，图数据库则把重点放在了高效的图查询和搜索上。
其次，RDF 以三元组的方式来存储数据而且不包含属性信息，但图数据库一般以属性图为基本的表示形式，所以实体和关系可以包含属性，这就意味着更容易表达现实的业务场景。
根据相关统计，图数据库仍然是增长最快的存储系统。相反，关系型数据库的增长基本保持在一个稳定的水平。


自然语言处理之知识图谱  https://blog.csdn.net/zourzh123/article/details/81011008

知识图谱实体链接：一份“由浅入深”的综述
------------------------------------------
太长了  放最后

https://zhuanlan.zhihu.com/p/100248426


.. image:: ../../_static/nlp/entity_linking.png
    :align: center
