.. knowledge_record documentation master file, created by
   sphinx-quickstart on Tue July 4 21:15:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

******************
推荐系统
******************


学习资料
===================
【推荐系统 python】推荐系统从入门到实战 https://www.youtube.com/watch?v=jNe1X6L1DyY&list=PLCemT-oocgalODXpQ-EP_IfrrD-A--40h

整体概念
=====================

为什么需要推荐系统
--------------------------------------------------------
| 1）信息过载(information overload)问题日益严重
| 2）人找喜欢的物品、资讯变得越来越困难
| 3）新的产品想脱颖而出、得到关注，亦不容易
 
推荐系统解决的问题
--------------------------------------------------------
| 1）人与物的精确匹配，从人找信息，转变为信息找人
| 2）帮助减少马太效应和长尾效应的影响


| 马太效应：产品中热门的东西会被更多人看到，热门的东西会变得更加热门，而冷门的东西更加冷门。
| 长尾理论：某些条件下，需求和销量不高的产品所占据的市场份额，可以和主流产品的市场份额相比。
 
.. image:: ../../_static/recommend/longtail.png
	:align: center
	:width: 300
	
	
推荐系统与搜索引擎的不同
--------------------------------------------------------

| **搜索引擎-人找资讯**
| 搜索引擎就是人找信息的经典情况，但是搜索出来的结果非常充分的体现了马太效应，就是越热门越靠前，没有体现个性化需求。
 
 
 
| **推荐系统-资讯找人**
| 1）事实上，每一个人的品味和偏好都并非和主流人群完全一致，当我们发现得越多，我们就越能体会到我们需要更多的选择。
| 2）如果说搜索引擎体现着马太效应的话，那么长尾理论则阐述了推荐系统发挥的价值。



从用户层面推荐系统的作用
--------------------------------------------------------
1）推荐系统能够满足用户对信息的个性化需求

推荐系统在个性化方面的运作空间要大得多，以“推荐好看的电影”为例，一百个用户有一百种口味，并没有一个“标准”的答案，推荐系统可以根据每位用户历史上的观看行为、评分记录等生成一个对当前用户最有价值的结果，这也是推荐系统有独特魅力的地方。

2）推荐系统满足用户难以用文字表述的需求

| •	用户天然都是愿意偷懒的，不愿意输入过多文字去精确表达自己的需求
| •	搜索引擎对语义的理解目前还无法做到足够深入
| •	推荐系统通过标签设置（页面上选择喜欢的标签），加上与用户的交互（筛选、排序、点击等），不断积累和挖掘用户偏好，可以将这些难以用文字表达的需求良好的满足起来
 
知识图谱在推荐系统中的作用
--------------------------------------------------------
知识图谱旨在描述真实世界中存在的各种实体或概念及其关系,其构成一张巨大的语义网络图，节点表示实体或概念，边则由属性或关系构成，知识图谱在推荐系统中能够起如下作用：
 
**精确性**：知识图谱为物品引入了更多的语义关系，可以深层次地发现用户兴趣。

.. image:: ../../_static/recommend/re_prec.png
	:align: center
	:width: 400

**多样性**：通过知识图谱中不同的关系链接种类，有利于推荐结果的发散。

.. image:: ../../_static/recommend/re_mul.png
	:align: center
	:width: 400
	
	
**可解释性**：知识图谱可以连接用户的历史记录和推荐结果，从而提高用户对推荐结果的满意度和接受度，增强用户对推荐系统的信任。

.. image:: ../../_static/recommend/re_expl.png
	:align: center
	:width: 400


推荐系统分类
--------------------------------------------------------
基于内容的推荐、协同过滤的推荐、混合的推荐

| 除此之外，还有
| 4）基于规则的推荐：这类算法常见的比如基于最多用户点击，最多用户浏览等，属于大众型的推荐方法，在目前的大数据时代并不主流。
| 5）基于人口统计信息的推荐：这一类是最简单的推荐算法了，它只是简单的根据系统用户的基本信息发现用户的相关程度，然后进行推荐，目前在大型系统中已经较少使用。


基于内容的推荐
=====================
基于内容的推荐 Content-based Recommendation

根据物品或内容的元数据，发现物品或内容的相关性，然后基于用户以前的喜好记录推荐给用户相似的物品，如图所示：

.. image:: ../../_static/recommend/cb.png
	:align: center
	:width: 400

用户喜欢A，因为A和C都有相同的类型（爱情，浪漫），所以把类似A的C推荐给用户。

基于内容的推荐只考虑了对象的本身性质，将对象按标签形成集合，如果你消费集合中的一个则向你推荐集合中的其他对象。

基于内容的推荐，依靠的是内容本身的相似性，比如把文本进行词袋表征，变成k维的向量，可以计算物品的相似度。
由于基于物品本身的文本或图像特征，没有冷启动问题，但是一般效果较差，因为很难在内容特征中提取用户偏好级别的内容相似性，实践中会发现，
你觉得计算出来的物品相似非常好，但是线上效果却很差。



协同过滤的推荐
=========================
协同过滤(Collaborative Filtering)作为推荐算法中最经典的类型，包括在线的协同和离线的过滤两部分。所谓在线协同，就是通过在线数据找到用户可能喜欢的物品，
而离线过滤，则是过滤掉一些不值得推荐的数据，比比如推荐值评分低的数据，或者虽然推荐值高但是用户已经购买的数据。

一般来说，协同过滤推荐分为三种类型。第一种是**基于用户**(user-based)的协同过滤，第二种是**基于项目**(item-based)的协同过滤，第三种是**基于模型**(model based)的协同过滤。

简单比较下基于用户的协同过滤和基于项目的协同过滤：基于用户的协同过滤需要在线找用户和用户之间的相似度关系，计算复杂度肯定会比基于基于项目的协同过滤高。
但是可以帮助用户找到新类别的有惊喜的物品。而基于项目的协同过滤，**由于考虑的物品的相似性一段时间不会改变，因此可以很容易的离线计算**，准确度一般也可以接受，
但是推荐的多样性来说，就很难带给用户惊喜了。一般对于小型的推荐系统来说，基于项目的协同过滤肯定是主流。但是如果是大型的推荐系统来说，则可以考虑基于用户的协同过滤，
当然更加可以考虑我们的第三种类型，基于模型的协同过滤。

基于模型(model based)的协同过滤是目前最主流的协同过滤类型了，我们的一大堆机器学习算法也可以在这里找到用武之地。




论文阅读
=====================


总体
----------------
| DeepCTR综述：深度学习用于点击率预估
| https://mp.weixin.qq.com/s/atP3uq8GgAQS9rIeQpa64w

| 互联网大厂CTR预估前沿进展
| https://mp.weixin.qq.com/s/B2GNzNfPqcY2_OxPR2aRng


| 算法大佬看了流泪，为什么这么好的CTR预估总结之前没分享(上篇)
| https://mp.weixin.qq.com/s/7Rer2qC54CbBYkPrNmWZRA
| 算法大佬看了流泪，为什么这么好的CTR预估总结之前没分享(下篇)
| https://mp.weixin.qq.com/s/WDvQlLjHrQE4zU3mdBMJfw

| 推荐系统技术演进趋势：排序篇
| https://mp.weixin.qq.com/s/gd7Y_cMVotnRcsdZSOcRcg
| 推荐系统技术演进趋势：重排篇
| https://mp.weixin.qq.com/s/YorzRyK0iplzqutnhEhrvw

| 万字长文梳理CTR点击预估模型发展过程与关系图谱
| https://mp.weixin.qq.com/s/qXK7EuBGby718OpcPxAaig
| 深度学习推荐系统、CTR预估工业界实战论文整理分享
| https://mp.weixin.qq.com/s/AJGX8kDrQkrIXPs2pzgn2A
| 机器学习和深度学习在CTR场景中的应用综述
| https://mp.weixin.qq.com/s/yIudTCaGQ8DH1ymlwUfZbQ

| CTR点击率预估论文集锦
| https://mp.weixin.qq.com/s/RVFxdCTpsWop3L8tQWaFjA
| 顶会中深度学习用于CTR预估的论文及代码集锦 (1)
| https://mp.weixin.qq.com/s/dSKKIjdtdZvU3kI5POzFEg
| 五大顶会2019必读的深度推荐系统与CTR预估相关的论文
| https://mp.weixin.qq.com/s/wIMNEXCF_PX1V0fLhNa-Cw
| KDD 2020关于深度推荐系统与CTR预估工业界必读的论文
| https://mp.weixin.qq.com/s/Twjw1N6RAV447BUEr2nUSw
| WSDM 2020关于深度推荐系统与CTR预估工业界必读的论文
| https://mp.weixin.qq.com/s/c0hPqwfbgdSKGvJwN5nX3A
| SIGIR 2020关于深度推荐系统与CTR预估相关的论文
| https://mp.weixin.qq.com/s/yN5_ZiowpCjP1Fg0_NHjfQ
| WWW 2020关于深度推荐系统与CTR预估相关的论文
| https://mp.weixin.qq.com/s/KITQYRFH6SD_2Y-f-2pyJA
| AAAI 2020关于深度推荐系统与CTR预估相关的论文
| https://mp.weixin.qq.com/s/43rv1YL9V0Dgfz_HId9OKw
| https://github.com/imsheridan/DeepRec
| https://github.com/shenweichen/DeepCTR

| SENet双塔模型：在推荐领域召回粗排的应用及其它
| https://mp.weixin.qq.com/s/1cvJUwXAsdoGA-lrp9RsFw

| 相关公众号：
| DataFun
| 炼丹笔记
| 小小挖掘机
| 深度学习
| 深度学习与NLP
| 深度传送门



感想
--------------------------------------------------------
| 1.	低阶特征相当重要。DCN里每次都留下低阶特征。
| 2.	是不是交叉相乘比mlp的效果好一些？
| 3.	点乘，元素积，相加相减，等等的特征交叉有优劣的说法吗
| FFM 
| Embedding分领域有什么好处

| 使用transformer？
| 平均池化可以优化？


FiBiNet  微博2019
-----------------------
使用Squeeze-Excitation network (Senet) 结构学习动态特征的重要性以及使用用双线性函数来更好的建模交叉特征


.. image:: ../../_static/recommend/fibinet_stru.png
	:align: center
	:width: 700

两个亮点。

| 1.把embedding后的向量经过了senet，相当于是加了每一维的attention。
| 2. 不是使用内积或者元素积（Hadamard product），他们提出了一种结合的方式，Bilinear-Interaction Layer

**亮点1：senet**

.. image:: ../../_static/recommend/senet.png
	:align: center
	:width: 300

| 有squeeze部分和excitation部分。  
| Squeeze部分相当于是压缩，可以max pooling或者ave pooling（之后adapt pooling？）。这篇文章里说，ave比原文的max效果好。有篇知乎文章说是因为避免被异常值带偏。
| Excitation部分相当于是权重，这里是两层mlp学习权重。

**亮点2：Bilinear-Interaction Layer**

.. image:: ../../_static/recommend/bilinear_inter.png
	:align: center
	:width: 500

| 内积是对应相乘
| 关于元素积

.. image:: ../../_static/recommend/hadamard.png
	:align: center
	:width: 500

感觉.....这种乘法和向量内积的区别，在于最后没有把3和8加起来，保留程度更高一些。

.. image:: ../../_static/recommend/inn_product.png
	:align: center
	:width: 400

Bilinear-Interaction Layer这个对于我们不太适用?因为相当于是要学n^2个权重。如果维度高了以后增加了很多计算成本。有评论也说这个复杂度有点高，换成内积速度快很多。

然后Combination Layer就是简单的拼接

.. image:: ../../_static/recommend/fibi_combination.png
	:align: center
	:width: 400


**实验结果数据分析**

测评Bilinear-Interaction Layer的效果

.. image:: ../../_static/recommend/fibi_bilinear_result.png
	:align: center
	:width: 400

00 01 这种指的是在两个embedding层后面接双线性层（00代表都不接，01代表SE-embedding的后面接，11代表都接以此类推）。
感觉看起来Bilinear-Interaction Layer的效果并没有提升多少。他自己写说在senet后面用这个效果稍好一些。


文章中还提到了Bilinear-Interaction Layer的三种拼接方式，看起来all的方式会好些。提升明显吗？但是计算量会上来。

.. image:: ../../_static/recommend/fibi_bilinear.png
	:align: center
	:width: 400


.. image:: ../../_static/recommend/fibi_bilinear_3ways_result.png
	:align: center
	:width: 300

至于后面DNN层的影响，


.. image:: ../../_static/recommend/fibi_dnn_result.png
	:align: center
	:width: 550

Ablation study

.. image:: ../../_static/recommend/fibi_Ablation_study.png
	:align: center
	:width: 300




DCN V2 
-------------------
https://zhuanlan.zhihu.com/p/353223660


.. image:: ../../_static/recommend/DCN_V2.png
	:align: center
	:width: 900