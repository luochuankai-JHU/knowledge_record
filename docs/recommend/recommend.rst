.. knowledge_record documentation master file, created by
   sphinx-quickstart on Tue July 4 21:15:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

******************
推荐系统
******************

工作总结
======================

`内部链接 <https://github.com/luochuankai-JHU/work_summary/blob/main/work_exp_v20240216.md>`_ 


学习资料
===================
【推荐系统 python】推荐系统从入门到实战 https://www.youtube.com/watch?v=jNe1X6L1DyY&list=PLCemT-oocgalODXpQ-EP_IfrrD-A--40h

！很多经典网络结构的代码实现  https://github.com/shenweichen/DeepCTR-Torch/tree/master


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
--------------------------------------------------------
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
--------------------------------------------------------
协同过滤(Collaborative Filtering)作为推荐算法中最经典的类型，包括在线的协同和离线的过滤两部分。所谓在线协同，就是通过在线数据找到用户可能喜欢的物品，
而离线过滤，则是过滤掉一些不值得推荐的数据，比比如推荐值评分低的数据，或者虽然推荐值高但是用户已经购买的数据。

一般来说，协同过滤推荐分为三种类型。第一种是**基于用户**(user-based)的协同过滤，第二种是**基于项目**(item-based)的协同过滤，第三种是**基于模型**(model based)的协同过滤。

简单比较下基于用户的协同过滤和基于项目的协同过滤：基于用户的协同过滤需要在线找用户和用户之间的相似度关系，计算复杂度肯定会比基于基于项目的协同过滤高。
但是可以帮助用户找到新类别的有惊喜的物品。而基于项目的协同过滤，**由于考虑的物品的相似性一段时间不会改变，因此可以很容易的离线计算**，准确度一般也可以接受，
但是推荐的多样性来说，就很难带给用户惊喜了。一般对于小型的推荐系统来说，基于项目的协同过滤肯定是主流。但是如果是大型的推荐系统来说，则可以考虑基于用户的协同过滤，
当然更加可以考虑我们的第三种类型，基于模型的协同过滤。

基于模型(model based)的协同过滤是目前最主流的协同过滤类型了，我们的一大堆机器学习算法也可以在这里找到用武之地。





NDCG : Normalized Discounted Cumulative Gain(归一化折损累计增益)
---------------------------------------------------------------------------
https://zhuanlan.zhihu.com/p/371432647

演变过程： G -> CG -> DCG -> NDCG

**Cumulative Gain: Gain的累加**

.. figure:: ../../_static/recommend/cg.png
	:scale: 70%

**Discounted Cumulative Gain**: 考虑排序顺序的因素，使得排名靠前的item增益更高，对排名靠后的item进行折损。

相当于乘上了一个随着index的折损系数。DCG在CG的基础上，给每个item的相关性除以log2(i+1)，i越大，log2(i+1)的值越大，相当于给每个item的相关性打个折扣，item越靠后，折扣越大。

相当于乘上 1/log(i + 1)

.. figure:: ../../_static/recommend/dcg.png
	:alt: DCG计算举例
	:scale: 50%



**NDCG(Normalized DCG): 归一化折损累计增益**

DCG基础上做了个归一化。所除的base 就是最佳情况下的 IDCG i means ideal 

解决每个result的list长度不一致导致无法比较的问题。都化成0-1范围的百分比


.. math::

  NDCG = \frac{DCG}{IDCG}


A/B testing
=====================
学习资料
---------------
https://www.youtube.com/watch?v=DUNk4GPZ9bw  A/B Testing in Data Science Interviews by a Google Data Scientist | DataInterview    这个说的挺好的。是一个基本的入门介绍

https://www.1point3acres.com/bbs/thread-643203-1-1.html

https://www.1point3acres.com/bbs/thread-809180-1-1.html

https://www.1point3acres.com/bbs/thread-945516-1-1.html



一些经验，需要消化
--------------------------------------------
主要看了udacity这个，但是只能说这个是入门级的AB test，很基础但是也很重要，一定要看明白，不明白重复看直到明白了，然后就可以进入第二重境界啦，看高深的AB test。onsite面试过程中AB test环节会问的很深入，基本不会出现udaciy那么简单的问题。
AB test 一般的套路就是 

understand business goal -> define metrics -> hypothesis -> design test plans, sample size? Duration? Regions for AB testing? -> 
launch experiment -> sanity check and analyze result -> conclusion/suggestion

.. image:: ../../_static/recommend/abtest2.png


[url=https://classroom.udacity.com/courses/ud257]https://classroom.udacity.com/courses/ud257[/url]
Notes：[url=http://rpubs.com/superseer/ab_testing]http://rpubs.com/superseer/ab_testing[/url]

AB test & hypo testing：也是看博客了，你会发现AB test真的是水很深，比如control和test的samples size是90% 和10% 分布，同时测10个metric而不是一个，又要改什么？AB 测试需要注意的事项和assumption是什么？ novelty effect是隐藏在这部分很重要的考点，怎么消除？违反了什么assumption？等等。加几个我平时看AB test的网站：
[url=https://towardsdatascience.com]https://towardsdatascience.com[/url]
[url=https://medium.com]https://medium.com[/url]
[url=https://www.optimizely.com]https://www.optimizely.com[/url]
这个帖子不错~
[url=https://towardsdatascience.com/a-summary-of-udacity-a-b-testing-course-9ecc32dedbb1]https://towardsdatascience.com/a-summary-of-udacity-a-b-testing-course-9ecc32dedbb1[/url]
machine learning
主要就是看面经了。自己也总结了些问题，希望能帮到大家，都无偿分享给大家了，攒人品。
[url=https://rpubs.com/JDAHAN/172473]https://rpubs.com/JDAHAN/172473[/url]


https://zhuanlan.zhihu.com/p/40919260  非统计专业？5分钟搞懂如何计算A/B测试样本量

https://zhuanlan.zhihu.com/p/565539453?utm_id=0  数据分析/数据科学 AB test常考题及答案

udacity a/b testing 笔记
-------------------------------------
不是所有的事情都适合于ab testing。比如 1：在测试一个新版本的时候，有的老用户可能就是会讨厌任何改变，但是有的用户就是会很喜欢体验新东西  2. 有的改变要很久才能有体现，比如房屋租赁推荐，人们可能要好几年才会回到这个网站来再次租赁

当A/B测试无效时，我们可以：

| 分析用户活动日志
| 进行回顾性分析
| 进行用户体验研究
| 焦点小组和调查
| 用户打分

以这个教学网站 Audacity为例

弄清楚 funnel 漏斗

.. image:: ../../_static/recommend/abtest3.png
	:width: 400

从用户进来网页，到搜索，到点击，到购买这节课

hypothesis: changing the buttom from orange to pink will increase user willing to buy courses and engagement

which metrics to choose(如何衡量？)

如果选择用户完成课程的比例，这个不好，因为太久了

所以，根据funnel, 选择用户看见这个页面后的点击率是合适的，特别是unique person的点击率

so, updated-hypothesis: changing the buttom from orange to pink will increase user's click-though probility

如果把点击率看成一个二项分布

m- mean 均值很好计算

σ - standard deviation 标准差 就是 sqrt (p(1-p) / N)

所以，离mean 举例 2σ之外的就是超出95%的置信区间了

null hypothesis and alternative hypothesis
零假设和对立假设

第一种简称Ho，意思是实验完全没带来任何变化

第二种是Ha，意思是实验有变化（好or坏）

这里插入那个截图！！！如何计算的



size V.S. power trade-off

Statistical power helps us find how many samples do we need to have statistical significance.  这里需要再看看！！


.. image:: ../../_static/recommend/abtest4.png

Pooled Standard Error

.. image:: ../../_static/recommend/abtest5.png

.. image:: ../../_static/recommend/abtest6.png

梳理Statistical Power和Significance Level  
----------------------------------------------------------------

首先，A/B测试包含两个假设：

原假设（Null hypothesis, 也叫H0）：我们希望通过实验结果推翻的假设。在我们的例子里面，原假设可以表述为“红色按钮和绿色按钮的点击率一样”。

备择假设（Alternative hypothesis, 也叫H1）：我们希望通过实验结果验证的假设。在我们的例子里面，可以表述为“红色按钮和绿色按钮的点击率不同”。

四种情况

.. image:: ../../_static/recommend/abtest1.png
	:width: 600

第一类错误（Type I error），用α表示。就是Significance Level。

第二类错误（Type II error），用β表示 β = 1 - power。


一般来说，第一类错误α不超过5%，第二类错误β不超过20%。也就是说，Significance Level = 5%。Statistical Power = 1 -β = 80%。


Minimum Detectable Effect  顾名思义，这个参数衡量了我们对实验的判断精确度的最低要求。

在工作中，这个参数的选定往往需要和业务方一起拍板。在我们的实验中，我们选定Minimum Detectable Effect=5%。这意味着，如果绿色按钮真的提高了点击率5个百分点以上，我们希望实验能够有足够把握检测出这个差别。
如果低于5个百分点，我们会觉得这个差别对产品的改进意义不大（可能是因为点击率不是核心指标），能不能检测出来也就无所谓了。

常考问题
---------------------

第一类错误 vs 第二类错误
-----------------------------------
面试官问：你怎么理解 AB 测试中的第一、二类错误？

第一类错误和第二类错误，AB 测试中最常提到的两个概念，也是统计学中比较容易搞混的两个概念。

举例理解，第一类错误代表我们误认为一个垃圾东西能带来收益，第二类错误是我们误认为一个好东西不能带来收益

往往在实际的工作中，第一类错误是我们更加不能接受的。换句更直白的话说，就是我们宁愿砍掉几个好的产品，也绝对不能让一个坏的产品上线。因为通常是一个坏的产品上线会给用户的体验带来极大的不好的影响，而这个不好的影响会非常大程度的影响到我们的日活以及留存。你要知道在现实生活中，我们把留存或者把日活提升一个百分点都已经是一个非常了不起的优化了，但是通常要实现 1%的留存率。人都需要花费很长时间，也很大精力，但是你如果想要留存下降一个百分点的话，可能就是一瞬间的事情。所以第一类错误通常是我们在实际工作当中所更加不能接受的。

统计显着=实际显着？
------------------------------
面试官问：如果你发现 AB 测试的结果在统计上来说是显着，但是在实际中却不显着，这是为什么？

这个可能的原因是我们在 AB 测试当中所选取的样本量过大，导致和总体数据量差异很小，这样的话即使我们发现一个细微的差别，它在统计上来说是显着的，在实际的案例当中可能会变得不显着了。
举个栗子，对应到我们的互联网产品实践当中，我们做了一个改动，APP 的启动时间的优化了 0.001 秒，这个数字可能在统计学上对应的 P 值很小，也就是说统计学上是显着的，
但是在实际中用户 0.01 秒的差异是感知不出来的。那么这样一个显着的统计差别，其实对我们来说是没有太大的实际意义的。所以统计学上的显着并不意味着实际效果的显着。


AB 测试效果统计上不显着？
---------------------------------------
面试官问：如果你发现你在 AB 测试当中所选取的指标在统计上来说都是不显着的，你该怎么去判断这个实验的收益？

对于这种情况，我们所选取的一种通用的方式是将这个指标去拆分成每一天去观察。如果指标的变化曲线每一天实验组都高于对照组，即使他在统计上来说是不显着的，
我们也认为在这样一个观测周期内，实验组的关键指标表现是优于对照组的，那么结合这样一个观测，我们最终也可以得出这个优化可以上线的结论。


实验组优于对照组就能上线？
------------------------------------------
面试官问：如果你在 AB 测试中发现实验组核心指标明显优于对照组，那这个优化就一定能够上线吗？

不一定。**we need to consider the tradeoff** 一个改动肯定是有好处有坏处的。

举个例子，比如说有的时候我们想要提升产品的视觉展现效果。但是这种优化可能是以用户等待内容展现的时间作为代价来进行提升的。所以一个方面的优化可能会导致另一个方面的劣化。
在做这个优化的时候，可能会对其他部门产生一些负向的影响，进而导致公司收入的下降。 所以我们在进行 AB 测试的时候，必须要综合评估所有方面的一些指标变动，
同时对于收益和损失来做一个评估，才能确认这个优化可以最终上线。



AB 测试的其他变式考法
-----------------------------------------------
你会如何去验证这个数字的显着性？
你的分析结果带来了一些显着的收益，你会是如何验证这些收益？



面试设计AB test的时候，一定要先明确 商业目标是什么
--------------------------------------------------------


ABtest有什么缺点？
---------------------------
场景受限

应用于短期行为，不适用于需要很长时间才能完成的测试

需要的用户人数较多，要有足够的样本量



















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


| 
| 

感想
--------------------------------------------------------
| 1.	低阶特征相当重要。DCN里每次都留下低阶特征。 很多模型都有类似resnet的结构保留低阶特征
| 2.	是不是交叉相乘比mlp的效果好一些？
| 3.	点乘，元素积，相加相减，等等的特征交叉有优劣的说法吗

关于相加减和乘机，看了 https://zhuanlan.zhihu.com/p/50426292

.. image:: ../../_static/recommend/fm_second_cross.png
	:align: center
	:width: 700

也许加减可以避免有一边为零导致相乘为零的情况？不知道是不是这个出发点

| FFM 
| Embedding分领域有什么好处

| 使用transformer？
| 平均池化可以优化？

senet

在特征上添加attention等权重

选取更多特征  （视频播完率等等）

做一些数据增强，比如一个高活用户，可以随机遮盖一些信息

通过他看了什么作者 继续推荐这个作者

matchnet把模型分开训练？ 分成低活人群的和高活人群的两个模型

dropout？一些特征随机置零  也算数据增强，沈老板关注

学习率warm up

BN 和 layer norm？

获取gr历史，一个月前点击的物料，取最相似  兴趣点返厂


| 关于离散值和连续值
| https://juejin.cn/post/6856021107054903304
| https://www.zhihu.com/question/31989952

.. image:: ../../_static/recommend/id_dense_disti.png
	:align: center
	:width: 700

| 
| 


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

| 笔记：
| 关于senet_ratio

.. image:: ../../_static/recommend/senet_ratio.png
	:align: center
	:width: 500

**亮点2：Bilinear-Interaction Layer**

.. image:: ../../_static/recommend/bilinear_inter.png
	:align: center
	:width: 500

| 内积是对应相乘
| 关于元素积(哈达玛积)

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


| 
| 


DCN V2 
-------------------
https://zhuanlan.zhihu.com/p/353223660

还发现一个好东西！很多经典网络结构的代码实现  https://github.com/shenweichen/DeepCTR-Torch/tree/master

我这里拾人牙慧一下。直接复制粘贴

DCN-V2(也称DCN-M  matrix)是在之前DCN-V1(也称DCN-V  vector)的基础上做了升级

先复习一下DCN-V1
````````````````````````

.. image:: ../../_static/recommend/DCNV1.png

原始数据进来后，sparse features先embedding然后和dense features拼接在一起，作为输入

右边的 deep network 就是简单的DNN

左边的cross network的核心思想是更高效地实现显式特征交叉，每一层的计算如下： 

.. image:: ../../_static/recommend/dcn-v1-cross.png

.. image:: ../../_static/recommend/dcn-v1-cross-frame.png


x\ :sub:`0`,  x\ :sub:`l`,  w\ :sub:`l`, b\ :sub:`l` 都是d维的列向量，形状是(d,1)。x\ :sub:`0` * x\ :sub:`l` T * w\ :sub:`l` 的形状是(d,1) * (1,d) * (d,1)=(d,1)，
与 x\ :sub:`l` 一致。cross网络每一层仅增加2d个参数（ w\ :sub:`l`, b\ :sub:`l`），整体参数量相比DNN是少得多的。


DCN-V2的改进
````````````````````````
DCN中cross网络的参数是向量，DCN-M中换成了矩阵来提高表达能力、方便落地。DCN-M是指“DCN-matrix” ，原来的DCN在这里称为DCN-V（“DCN-vector”）。

.. image:: ../../_static/recommend/dcnv2.png

x\ :sub:`l+1` = x\ :sub:`0` ⊙ (W\ :sub:`l` * x\ :sub:`l` + b\ :sub:`l`) + x\ :sub:`l`

⊙是哈达玛积（Hadamard product），就是对应元素相乘，这个博客页面多次介绍过


一句话介绍：

to get next layer, element-wise multip the initial input x0 with the linear transformation (wx+b) of current layer xl, and then adding xl itself.

.. image:: ../../_static/recommend/dcn-v2.png

结构上有并行和串行两种，论文中说效果各有优劣。我们经过实验得出适合我们的是串行












AFN
---------------------------------------------------------------------------------
Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions

一篇AAAI20的论文。主要特色是引入了对数。

**论文摘要**

| 目前的fm方法是基于二阶交叉或者高阶交叉。这样会有两个问题：
| 1.他们必须在高阶交叉特征的表达能力和计算成本之间进行权衡，从而导致次优预测。
| 2.枚举所有交叉特征，包括不相关的特征，可能会引入噪声特征组合，从而降低模型性能。

本文提出的AFN 可以从数据中学习任意阶的特征。核心思想是引入对数mic变换，将特征对数化，再去做交叉运算。这样能将特征组合中每个特征的幂转换为带系数的乘法。


**Introduction部分**

| 提出两个问题：
| 1. 模型该使用多高阶的特征？因为使用上高阶特征是会对结果有益的，但是会带来更多的计算成本。
| 2.哪些交叉的特征是有用的

**Background部分**

这里先来对论文里出现的符号做个总结：
xi 是第i个feature field表示的特征向量（没有做embedding）

ei=vi*xi
ei是做了embedding后的特征向量

这是普通的二阶交叉

.. image:: ../../_static/recommend/afn_second_order.png
	:align: center
	:width: 400

这是普通的高阶交叉

.. image:: ../../_static/recommend/afn_high_order.png
	:align: center
	:width: 400

目前的交叉都是限定好了阶数。

这里借鉴了Logarithmic Neural Network (LNN)的思想。关于lnn

.. image:: ../../_static/recommend/afn_lnn.png
	:align: center
	:width: 550
 
对数化
LNN 的思想是将输入转换为对数空间，将乘法转换为加法，将除法转换为减法，将幂转换为常数



**Afn结构**

.. image:: ../../_static/recommend/afn_afn_structor.png
	:align: center
	:width: 800

| 输入有两点值得注意：
| 1.由于对数里面不能有负数，所以embedding层的内容都是正数
| 2.对数里是0的数字换成了一个小正数

（6）中的公式在对数转换层会变成

.. image:: ../../_static/recommend/afn_7_formular.png
	:align: center
	:width: 500

.. image:: ../../_static/recommend/afn_7_formular_explain.png
	:align: center
	:width: 600

举例说明的话，如果想看二阶交叉，只保留e1和e2。其他的权重置零。


DNN层

在fm后面串接了dnn，激活函数选的relu


**实验结果**

.. image:: ../../_static/recommend/afn_exp_result.png
	:align: center
	:width: 800

ensemble的方式的确有用
CIN值得关注

 
在使用ensemble的时候，AFN和dnn是分开训练的，embedding空间也没有共享。

.. image:: ../../_static/recommend/afn_ensemble.png
	:align: center
	:width: 500

**Ablation study**

.. image:: ../../_static/recommend/afn_ablation.png
	:align: center
	:width: 500

| A。没看懂这里指的是什么
| B。后面接一层dnn能有效提升，再多了意义不大
| C。dnn的宽度调节起来有影响。过深或者过浅都不合适。具体数据要结合业务。



| 
| 



Facebook Que2Search
---------------------------------------------------------------------------------
Que2Search: Fast and Accurate Query and Document Understanding for Search at Facebook

一篇Facebook的论文。满满的工业风，真正来说，技术上的创新点不太显眼，但是各种工程落地的细节很详实。

**0.Abstract**

.. image:: ../../_static/recommend/que_abs.png
	:align: center
	:width: 400

| 这个部分是介绍了一下他们的query2search已经应用在了facebook marketplace search。这是个类似淘宝的业务，用户搜一个东西，他们展示个性化的商品。

.. image:: ../../_static/recommend/que_hat.png
	:align: center
	:width: 300

| 他们这里"明目张胆"的把公司的名字挂上去，我们之后写文章也可以？


**1.INTRODUCTION**

介绍各个模块的发展历程

| 值得注意的是，他这里直接写的是Que2Search is trained on weakly-supervised datasets and achieves state-of-theart performance for product representation compared to previous baselines at Facebook
| 所以他的benchmark就是自己原本的基线......
| 他这里写的面临的挑战也很..有趣。可能这就是工业界论文的写法吧

.. image:: ../../_static/recommend/que_chanllenge.png
	:align: center
	:width: 500

| 一个是数据集的噪声....哪个数据集没噪声啊....特被是工业界的
| 多语言。这个比我们复杂一些
| multi-modalities 这有啥好写的
| 延迟要求。

**2.RELATED WORK**

| 没啥太多亮点。
| 有个地方提到了Siamese networks

.. image:: ../../_static/recommend/que_siamese.png
	:align: center
	:width: 600

还提到了 early fusion。这个也是我们可以尝试的方向

**3.MODELING**
这里提到了使用更难的负样本，这也是我们尝试的方向。他这里的添加更难负样本的方式还不需要改变训练数据，后文会讲。

3.1 Model architecture

| 这里提到了EmbeddingBag 

.. image:: ../../_static/recommend/que_embbag.png
	:align: center
	:width: 600

然后就是大家最关心的整体框架


.. image:: ../../_static/recommend/que_framework.png
	:align: center
	:width: 700

query侧，query的3-gram做了一个emb，county做了一个emb，query本身通过XLM做了emb，然后是attention fusion，相当于是对三种输入加了attention。

在doc侧，标题和摘要各通过xlm做了emb，title的3-gram做了emb，摘要的3-gram的emb和图片（已经pretrained）。也是有attention fusion。最后query的emb和doc的emb做余弦相似度。

注意，他这里通过XLM获取文字emb的方式也是通过 [CLS] 位置的emb来代替整句的emb

文中提到，simple attention fusion效果比直接拼接要好

然后还使用了dropout (rate = 0.1) ，gradient clipping of 1.0 和 early stopping with a patience of 3 epochs


.. image:: ../../_static/recommend/que_multitask1.png
	:align: center
	:width: 500


这个地方提到了多任务学习，我不了解，可以参考另一篇解读的


.. image:: ../../_static/recommend/que_multitask2.png
	:align: center
	:width: 600

3.2 Training

本篇的训练是分两个阶段的。

他们是这样定义正样本的（因为人工标注的样本量太少，需要借助海量的用户弱监督行为数据）

.. image:: ../../_static/recommend/que_positive_sample.png
	:align: center
	:width: 500

关于正负样本，他们是使用的list-wise。在一个batch中，假设q从1到i，doc从1到i。那么对于任意的qj，其实只有第j个（query和doc）是匹配上的。所以对于第j个，只有qj和dj才是正样本，qj和其他不为j的d都是负样本。这样会把问题转化为 multi-class classification problem


.. image:: ../../_static/recommend/que_sample_matrix.png
	:align: center
	:width: 500

他们还使用了scaled multi-class cross-entropy loss


.. image:: ../../_static/recommend/que_scale_softmax.png
	:align: center
	:width: 500

这样可以拉大正负cos直接的exp，加快收敛

他们还尝试了Symmetrical Scaled Cross Entropy Loss 。本来是q找d，对称就是再加上d找q

.. image:: ../../_static/recommend/que_symmetrical_loss.png
	:align: center
	:width: 500

作者表示，该损失函数并没有对query to document的双塔模型有所增益。但是在另外的一个document-to-document检索场景中，有2%的ROC AUC增益

3.3 Curriculum Training

这个是第二阶段的训练。使用的是harder negative examples。获得了absolute over 1% ROC AUC 增益

.. image:: ../../_static/recommend/que_2train_auc.png
	:align: center
	:width: 500

关于样本的生成，这个地方说的很清楚

.. image:: ../../_static/recommend/que_hard_sample.png
	:align: center
	:width: 500

在阶段一中，qi di是指定的正样本，但是在这一组list中，负样本中会有一个score最大的dnqi。这个可以视为最难的负样本。（
感觉对应到我们的业务就是 高相关里面再找高点展样本？）然后这样学习的是一个三元组 (qi, di, dnqi)。这边部分的loss是margin rank loss 。
一开始这个curriculum training并不有效，后来发现要先在一阶段收敛了才行

| 3.4 Evaluation
| 3.5 Speeding up model inference
| 这两个部分没有啥好讲的

3.6 Fusion of different modalities

.. image:: ../../_static/recommend/que_modalities1.png
	:align: center
	:width: 600


.. image:: ../../_static/recommend/que_modalities2.png
	:align: center
	:width: 600

多模态融合这个不太了解，详情见另一篇解读


.. image:: ../../_static/recommend/que_modalities3.png
	:align: center
	:width: 600

3.7 Model Interpretability

3.7.1 Does XLM encoder add value to the query tower?

对于这个问题，作者用attention fusion的时候的权重来诠释的。因为他使用的是softmax激活函数，相当于各权重求和为1。
这样，计算得到XLM占比达到了0.64。除此之外，随着query的变长，模型会更加关注xlm。当query小于5个字时模型更关注n-gram。当字变多时几乎全部关注XLM

3.7.2 Feature Importance

这里探究特征重要度的方式和我们一样---feature ablation。就是对某特征随机置零或者置一个随机数，看auc下降多少。

.. image:: ../../_static/recommend/que_feature_imp.png
	:align: center
	:width: 400

这里document的groknet是预训练好的图片的vec。可以看出，在duc侧他们的图片占比是最高的


**4.SYSTEM ARCHITECTURE**

一些工程侧的部署。

也是分离线和在线计算。doc侧是计算好后入库，query侧因为时效性要求实时计算。doc侧计算好后的vec会随着模型更新而更新。


**5.ABLATION STUDIES**

.. image:: ../../_static/recommend/que_ablation.png
	:align: center
	:width: 500

后面的部分没有太多想说的。这里可以提一下

6.5 Search Ranking 

他们的排序其实也分为粗排和精排两部分。粗排是GBDT，精排是DLRM-like model 。在排序阶段是使用了Que2search的分数的。

6.6 Lessons from failures

这里他们总结了一下经验教训。

Precision matters:

放低阈值会带来不好的效果。他们认为这是由于召回和排序的不一致造成的。放开阈值后，排序模型无法处理更多的噪声数据。
这个和我们放开召回进粗排的量导致性能下降有类似之处。保持多阶段模型的连续性是另一个较大的话题。

这里有两篇相关的论文

Zhihong Chen, Rong Xiao, Chenliang Li, Gangfeng Ye, Haochuan Sun,and Hongbo Deng. 2020. ESAM: Discriminative Domain Adaptation with Non-Displayed Items to Improve Long-Tail Performance. arXiv preprint arXiv:2005.10545 (2020).

Bowen Yuan, Jui-Yang Hsia, Meng-Yuan Yang, Hong Zhu, Chih-Yao Chang, Zhenhua Dong, and Chih-Jen Lin. 2019. Improving ad click prediction by considering non-displayed events. In KDD.

只保证相关性远远不够。 

提高召排一致性的一种方法是直接将召回的相似性分数用在排序中。期望的结果是，召回引入的相关性差的内容，排序能够将其排在后面。
实际却不然，相关性的NDCG确实提升的，但是线上指标却下降了。 
This is possibly because the two-tower model is trained to optimize query-product similarity instead of optimizing engagement, 
while the GBDT model is more engagement focused.就算将双塔的输出作为排序模型的输入也不能很好的缓解这种现象


**7.CONCLUSION**

我们介绍了构建名为 Que2Search 的综合查询和产品理解系统的方法。 我们提出了关于多任务和多模式训练的创新想法，以学习查询和产品表示。 
通过 Que2Search，我们实现了超过 5% 的绝对离线相关性改进和超过 4% 的在线参与度，超过了最先进的 Facebook 产品底层系统。 
我们分享了我们在针对搜索用例调整和部署基于 BERT 的查询理解模型方面的经验，并在第 99 个百分位实现了 1.5 毫秒的推理时间。 
我们分享了我们的部署故事、部署步骤的实用建议，以及如何将 Que2Search 组件集成到搜索语义召回和排序阶段中。


**参考**

Que2Search: Fast and Accurate Query and Document Understanding for Search at Facebook

https://blog.csdn.net/chao_1083934282/article/details/120598266

https://zhuanlan.zhihu.com/p/415516966

| 
| 

广义随机森林GRF（Generalized Random Forests）
---------------------------------------------------------------------------------
论文阅读：广义随机森林GRF（Generalized Random Forests）论文笔记补充
https://zhuanlan.zhihu.com/p/599357475

**论文原文**

Generalized random forests （S. Athey, J. Tibshirani, S. Wager. Generalized random forests. Annals of Statistics, 47, no. 2, 1148--1178）

**说明**

这里只是补充一下，在大家看GRF的论文原文前，整理GRF的构建过程和基本思想，方便理解。按照论文顺序进行解读的文章，
https://zhuanlan.zhihu.com/p/589094281 
这篇文章已经说的非常好了，我在阅读的时候也是结合这篇解读来理解的。

**先温习一下随机森林**

GRF（Generalized Random Forests）其实是在随机森林的基础上进行了改进。那么我们先非常迅速的回顾一下随机森林，有助于我们对广义随机森林的理解。

.. image:: ../../_static/recommend/grf_rf.png
	:align: center
	:width: 600


随机森林是由多棵决策树构成。假设有10棵树，每棵树拿到的都是随机采样后的样本和随机采样后的特征（形成差异化，避免过拟合），
然后按照Gini等方法去分裂。在分类问题中，如果10棵树里面有8棵树预测出来是猫，那么预测结果是猫的概率就是8/10，
每棵树的投票权重都是一样的。

关于随机森林和决策树的其他知识点可以看我的博客 
https://knowledge-record.readthedocs.io/zh_CN/latest/machine_learning/machine_learning.html#id10

**广义随机森林总的来说，相对于随机森林有两点不同**

1.每棵树的权重有差异

2.分裂方法不同

我们展开来说

**每棵树权重有差异**

在上面提到的，10棵决策树来预测猫的问题中，我们可以感性的认知到，这10棵树的权重是可以有所不同的：由于在树的生长过程中进行了采样，获取的样本是不同的，获取的特征也是不同的，那么这些树是在预测问题上是有能力上的差异的。通俗的讲，应该是“牛逼”的树权重大，"辣鸡"的树权重小。那么，用什么指标来衡量这个权重呢？这是整个GRF的核心思想。这个问题其实很难。理论上是求解方程（2）的解

.. image:: ../../_static/recommend/grf_2.png
	:align: center
	:width: 450

直接求解方程（2）在低维度下是可以实现的，但是在高维度下会遇到计算成本过大的问题，论文是用森林加权的近似方法代替了核函数加权：使用权重 αi(x) 表示训练样本i与测试样本x的相似度，通过加权实现异质性估计。

既：在计算每棵树的权重的时候，计算的其实是测试样本X与这棵树B的相似性，记为αi(x)。

相似性的通俗理解：决策树在分裂的时候，是把所接收的数据逐渐分裂成各个叶子结点，那么如果测试样本能够落入那个叶子节点，则认为有相似性。更进一步，如果那个落入的叶子节点中训练样本很多，说明没划分完全，特异性低，那么权重就低；如果该叶子结点中训练样本少，说明划分的很完全了，精细化程度高，权重就高。具体可见公式（3）

.. image:: ../../_static/recommend/grf_3.png
	:align: center
	:width: 450

公式（3）的意思是：对于树的集合 (index b从1....B)，对于任意一棵树B ，分母Lb(x)指的是和测试样本x落入同一叶子节点的训练样本的数量。分子表示如果有落入则取1，没有则取0.

这样可以得到任意一棵树的暂时权重α_bi(x)，表示第i个训练样本和样本x落入同一个叶子节点的频率。之后再做一个归一化，求得每棵树的真正权重αi(x)。

**分裂方法不同**

由于计算相似性是用落入该叶子结点的样本数量来衡量的，所以文章要求局部特征空间内数据是同质的，分裂的时候需要按照最大化异质性的方法去分裂，而不是GINI等方式。具体细节可以看原文。

**总体构建过程**

.. image:: ../../_static/recommend/grf_build.png
	:align: center
	:width: 600

我来画个流程图解读一下（符号与上面的伪代码保持一致）

.. image:: ../../_static/recommend/grf_build_me.png
	:align: center
	:width: 600

整体流程已经整理好了，其他部分看论文原文就好


**参考**

Generalized random forests （S. Athey, J. Tibshirani, S. Wager. Generalized random forests. Annals of Statistics, 47, no. 2, 1148--1178）

巴拉巴拉：因果推断笔记 | 广义随机森林GRF（Generalized Random Forests） https://zhuanlan.zhihu.com/p/589094281

一般化隨機森林 (Generalized Random Forest)   https://taweihuang.hpd.io/2020/04/27/generalized-random-forest/

阙斌斌：generalized random forests笔记  https://zhuanlan.zhihu.com/p/397546177

论文笔记：Generalized Random Forests   https://blog.csdn.net/zyl_wjl_1413/article/details/125380173










Deep Neural Networks for YouTube Recommendations
----------------------------------------------------------------
没完全看明白 先放这里


**ABSTRACT**

youtube将推荐系统分成了两个部分：deep candidate generation model 和 deep ranking model.

然后文章还分享了一些经验

**1. INTRODUCTION**

YouTube面临的三大问题：

• Scale 

• Freshness

• Noise:由于稀疏性和各种不可观察的外部因素，YouTube 上的历史用户行为本质上很难预测。 我们很少获得用户满意度的基本事实，而是对噪声隐式反馈信号进行建模。

The paper is organized as follows: 

Section 2: a brief system overview

Section 3 describes the **candidate generation model** in more detail, including how it is trained and used to serve recommendations. Experimental results
will show how the model benefits from **deep layers of hidden units** and **additional heterogeneous signals**. 

Section 4 details the **ranking model**, including how classic **logistic regression** is modified to train a model predicting **expected watch time**
(rather than click probability). Experimental results will show that **hidden layer depth** is helpful as well in this situation. 

Section 5 presents our conclusions and lessons learned.


**2. SYSTEM OVERVIEW**

.. image:: ../../_static/recommend/YTB_framework.png


他们的物料称之为 video corpus

candidate generation model 会将million级别的corpus降低到百级别

然而，为了最终确定算法或模型的有效性，我们依靠通过现场实验进行 A/B 测试。 在现场实验中，我们可以测量点击率、观看时间以及许多其他衡量用户参与度的指标的细微变化。 
这很重要，因为实时 A/B 结果并不总是与离线实验相关。


**3. CANDIDATE GENERATION**

3.1 Recommendation as Classification

这里将推荐系统问题看成是 extreme multiclass classification

.. image:: ../../_static/recommend/YTB_problem.png

计算 watch w\ :sub:`t` at time t 的时候，观看 **videos i** (classes) from a **corpus V** based on a **user U** and **context C** 的概率

u ∈ R\ :sup:`N`
represents a high-dimensional “embedding” of the user :durole:`superscript

v\ :sub:`j` ∈ R\ :sup:`N` represent embeddings of each candidate video

R\ :sup:`N` 都是dense vector 

模型的任务是  学习 user embeddings u as a function of the user’s history and context that are useful for discriminating among videos with a softmax classifier