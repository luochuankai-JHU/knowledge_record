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

 

 

bert & transformer
=================

马上上线


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

https://cloud.tencent.com/developer/article/1558479

https://zhuanlan.zhihu.com/p/148656446

史上最全Transformer面试题

Transformer为何使用多头注意力机制？（为什么不使用一个头）
---------------------------------------------------------------

这个目前还没有公认的解释，本质上是论文原作者发现这样效果确实好。但是普遍的说法是，使用多个头可以提供多个角度的信息。


Transformer为什么Q和K使用不同的权重矩阵生成，为何不能使用同一个值进行自身的点乘？
-----------------------------------------------------------------------------------------

.. image:: ../../_static/nlp/self-attention.png
	:align: center

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

| 4. 为什么在进行softmax之前需要对attention进行scaled（为什么除以dk的平方根），并使用公式推导进行讲解
| 5. 在计算attention score的时候如何对padding做mask操作？
| 6. 为什么在进行多头注意力的时候需要对每个head进行降维？（可以参考上面一个问题）
| 7. 大概讲一下Transformer的Encoder模块？
| 8. 为何在获取输入词向量之后需要对矩阵乘以embedding size的开方？意义是什么？
| 9. 简单介绍一下Transformer的位置编码？有什么意义和优缺点？
| 10. 你还了解哪些关于位置编码的技术，各自的优缺点是什么？
| 11. 简单讲一下Transformer中的残差结构以及意义。
| 12. 为什么transformer块使用LayerNorm而不是BatchNorm？LayerNorm 在Transformer的位置是哪里？
| 13. 简答讲一下BatchNorm技术，以及它的优缺点。
| 14. 简单描述一下Transformer中的前馈神经网络？使用了什么激活函数？相关优缺点？
| 15. Encoder端和Decoder端是如何进行交互的？（在这里可以问一下关于seq2seq的attention知识）
| 16. Decoder阶段的多头自注意力和encoder的多头自注意力有什么区别？（为什么需要decoder自注意力需要进行 sequence mask)
| 17. Transformer的并行化提现在哪个地方？Decoder端可以做并行化吗？
| 18. 简单描述一下wordpiece model 和 byte pair encoding，有实际应用过吗？
| 19. Transformer训练的时候学习率是如何设定的？Dropout是如何设定的，位置在哪里？Dropout 在测试的需要有什么需要注意的吗？
| 20. 引申一个关于bert问题，bert的mask为何不学习transformer在attention处进行屏蔽score的技巧？