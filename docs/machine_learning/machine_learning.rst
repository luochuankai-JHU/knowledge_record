.. knowledge_record documentation master file, created by
   sphinx-quickstart on Tue July 4 21:15:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

******************
Machine_Learning
******************

ensemble
=====================

änˈsämbəl

集成学习ensemble分为bagging(减小方差)、boosting(偏差) 和 stacking(改进预测)

bagging和boost
---------------------
| Bagging算法是这样做的：每个分类器都随机从原样本中做有放回的采样，然后分别在这些采样后的样本上训练分类器，然后再把这些分类器组合起来。简单的多数投票一般就可以。其代表算法是随机森林。

| AdaBoosting方式每次使用的是全部的样本，每轮训练改变样本的权重。下一轮训练的目标是找到一个函数f 来拟合上一轮的残差。
当残差足够小或者达到设置的最大迭代次数则停止。Boosting会减小在上一轮训练正确的样本的权重，增大错误样本的权重。（对的残差小，错的残差大）

梯度提升的Boosting方式是使用代价函数对上一轮训练出的模型函数f的偏导来拟合残差。

| Bagging和Boosting的区别（面试准备）
| https://www.cnblogs.com/earendil/p/8872001.html

Bagging和Boosting的区别
------------------------------

1）样本选择上：

Bagging：训练集是在原始集中有放回选取的，从原始集中选出的各轮训练集之间是独立的。

Boosting：每一轮的训练集不变，只是训练集中每个样例在分类器中的权重发生变化。而权值是根据上一轮的分类结果进行调整。

2）样例权重：

Bagging：使用均匀取样，每个样例的权重相等

Boosting：根据错误率不断调整样例的权值，错误率越大则权重越大。

3）预测函数：

Bagging：所有预测函数的权重相等。

Boosting：每个弱分类器都有相应的权重，对于分类误差小的分类器会有更大的权重。

4）并行计算：

Bagging：各个预测函数可以并行生成

Boosting：各个预测函数只能顺序生成，因为后一个模型参数需要前一轮模型的结果。

bagging减少方差，boosting是减少偏差
-----------------------------------------------------------------
bagging是减少variance方差，而boosting是减少bias偏差？？？再补充

stacking
--------------------

.. image:: ../../_static/machine_learning/stacking.png
	:align: center
	
关于stacking 的使用
https://blog.csdn.net/Li_yi_chao/article/details/89638009
	

GBDT
-------------------------
| GBDT与Adboost最主要的区别在于两者如何识别模型的问题。Adaboost用错分数据点来识别问题，通过调整错分数据点的权重来改进模型。GBDT通过负梯度来识别问题，通过计算负梯度来改进模型。

| 推荐GBDT树的深度：6；（横向比较：DecisionTree/RandomForest需要把树的深度调到15或更高）

| 提升树与梯度提升树：利用损失函数的负梯度在当前模型的值，代替提升树算法中的残差的作用


| GBDT 台大林轩田老师讲解视频：
| https://www.youtube.com/watch?v=pTNKUj_1Dw8&list=PL1AVtvtzG0LYN-dOGPYyRrzzyI5fk_D4H&index=31

| 这个文字资料写的不错：
| http://www.52caml.com/head_first_ml/ml-chapter6-boosting-family/

| 看看这个博客 机器学习算法GBDT  https://www.cnblogs.com/bnuvincent/p/9693190.html 里面讲了：
| gbdt 的算法的流程？
| gbdt 如何选择特征 ？
| gbdt 如何构建特征 ？
| gbdt 如何用于分类？
| gbdt 通过什么方式减少误差 ？
| gbdt的效果相比于传统的LR，SVM效果为什么好一些 ？
| gbdt 如何加速训练？
| gbdt的参数有哪些，如何调参 ？
| gbdt 实战当中遇到的一些问题 ？
| gbdt的优缺点 ？

GBDT 基本思想是根据当前模型损失函数的负梯度信息来训练新加入的弱分类器，然后将训练好的弱分类器以累加的形式结合到现有的模型中。

截图选取一些部分

.. image:: ../../_static/machine_learning/gbdt.png
	:align: center

GBDT 如何用于分类？？？？


GBDT和随机森林异同
------------------------------
**GBDT和随机森林的相同点**

1、都是由多棵树组成

2、最终的结果都是由多棵树一起决定

**GBDT和随机森林的不同**

1、组成随机森林的树可以是分类树，也可以是回归树；而GBDT只由回归树组成 ？？？

2、组成随机森林的树可以并行生成；而GBDT只能是串行生成

3、对于最终的输出结果而言，随机森林采用多数投票等；而GBDT则是将所有结果累加起来，或者**加权**累加起来

4、随机森林对异常值不敏感，GBDT对异常值非常敏感

5、随机森林对训练集一视同仁，GBDT是基于权值的弱分类器的集成

6、随机森林是通过减少模型方差提高性能，GBDT是通过减少模型偏差提高性能


AdaBoost算法缺点
--------------------------
| 对异常点敏感
| 最终模型无法用概率来解释


Xgboost 为什么用二阶泰勒展开
------------------------------------------
| 1.能自定义损失函数。  为了可以设置任何可以二阶求导的损失函数，只要该损失函数二阶可导，都可以用泰勒展开式进行近似替代，实现形式上的"统一"
| 2.收敛速度更快。 一阶信息描述梯度变化方向，二阶信息可以描述梯度变化方向是如何变化的。


| 二阶信息本身就能让梯度收敛更快更准确。这一点在优化算法里的牛顿法里已经证实了。可以简单认为一阶导指引梯度方向，二阶导指引梯度方向如何变化。这是从二阶导本身的性质，也就是为什么要用泰勒二阶展开的角度来说的

| 收敛速度上有提升

| 知乎：最优化问题中，牛顿法为什么比梯度下降法求解需要的迭代次数更少？https://www.zhihu.com/question/19723347

| Xgboost https://juejin.im/post/5d2590e1e51d45106b15ffaa 这篇文章讲的不错


XGBoost与GBDT有什么不同
---------------------------------
| 除了算法上与传统的GBDT有一些不同外，XGBoost还在工程实现上做了大量的优化。总的来说，两者之间的区别和联系可以总结成以下几个方面。
| 1.	GBDT是机器学习算法，XGBoost是该算法的工程实现。
| 2.	在使用CART作为基分类器时，XGBoost显式地加入了正则项来控制模型的复杂度，有利于防止过拟合，从而提高模型的泛化能力。正则项里包含了树的叶子节点个数、每个叶子节点上输出的score的L2模的平方和
| 3.	GBDT在模型训练时只使用了代价函数的一阶导数信息，XGBoost对代价函数进行二阶泰勒展开，可以同时使用一阶和二阶导数。
| 4.	传统的GBDT采用CART作为基分类器，XGBoost支持多种类型的基分类器，比如线性分类器。
| 5.	传统的GBDT在每轮迭代时使用全部的数据，XGBoost则采用了与随机森林相似的策略，支持对数据进行采样。
| 6.	传统的GBDT没有设计对缺失值进行处理，XGBoost能够自动学习出缺失值的处理策略。

XGBoost处理缺失值
------------------------------
| 1）xgboost分别假设该样本属于左子树和右子树，比较两者分裂增益，选择增益较大的那一边作为该样本的分裂方向。
| 2）指定一个默认方向，比如在训练中没有缺失值而在预测中出现缺失，那么会自动将缺失值的划分方向放到右子树。

原是论文中关于缺失值的处理将其看与稀疏矩阵的处理看作一样。在寻找split point的时候，不会对该特征为missing的样本进行遍历统计，
只对该列特征值为non-missing的样本上对应的特征值进行遍历，通过这个技巧来减少了为稀疏离散特征寻找split point的时间开销。在逻辑实现上，为了保证完备性，
会分别处理将missing该特征值的样本分配到左叶子结点和右叶子结点的两种情形，计算增益后选择增益大的方向进行分裂即可。可以为缺失值或者指定的值指定分支的默认方向，
这能大大提升算法的效率。如果在训练中没有缺失值而在预测中出现缺失，那么会自动将缺失值的划分方向放到右子树。



决策树与这些算法框架进行结合所得到的新的算法：
----------------------------------------------------------------
| 1）Bagging + 决策树 = 随机森林
| 2）AdaBoost + 决策树 = 提升树
| 3）Gradient Boosting + 决策树 = GBDT

xgboost判断特征重要程度的三种指标
-------------------------------------------
掉包里面，get_fscore 有三种种评判特征重要程度的方法：

| ‘weight’ - the number of times a feature is used to split the data across all trees.
| ‘gain’ - the average gain of the feature when it is used in trees.
| ‘cover’ - the average coverage of the feature when it is used in trees.

| weight - 该特征在所有树中被用作分割样本的特征的次数。
| gain - 在所有树中的平均增益。
| cover - 在树中使用该特征时的平均覆盖范围。

lightGBM
--------------------
基本原理与XGBoost一样，只是在框架上做了一优化（重点在模型的训练速度的优化）。


lightGBM与XGboost对比
-----------------------------------
| 1、xgboost采用的是level-wise的分裂策略，而lightGBM采用了leaf-wise的策略，区别是xgboost对每一层所有节点做无差别分裂，可能有些节点的增益非常小，
对结果影响不大，但是xgboost也进行了分裂，带来了务必要的开销。 leaft-wise的做法是在当前所有叶子节点中选择分裂收益最大的节点进行分裂，如此递归进行，很明显leaf-wise这种做法容易过拟合，因为容易陷入比较高的深度中，因此需要对最大深度做限制，从而避免过拟合。

| 2、lightgbm使用了基于histogram的决策树算法，这一点不同与xgboost中的 exact 算法（tree_method 可以使用 hist参数），histogram算法在内存和计算代价上都有不小优势。
| 　　（1）内存上优势：很明显，直方图算法的内存消耗为(#data* #features * 1Bytes)(因为对特征分桶后只需保存特征离散化之后的值)，而xgboost的exact算法内存消耗为：(2 * #data * #features* 4Bytes)，因为xgboost既要保存原始feature的值，也要保存这个值的顺序索引，这些值需要32位的浮点数来保存。
| 　　（2）计算上的优势，预排序算法在选择好分裂特征计算分裂收益时需要遍历所有样本的特征值，时间为(#data),而直方图算法只需要遍历桶就行了，时间为(#bin)

| 3、直方图做差加速
| 一个子节点的直方图可以通过父节点的直方图减去兄弟节点的直方图得到，从而加速计算。

| 4、lightgbm支持直接输入categorical 的feature
| 在对离散特征分裂时，每个取值都当作一个桶，分裂时的增益算的是”是否属于某个category“的gain。类似于one-hot编码。

| 5、多线程优化

 

lightgbm哪些方面做了并行
------------------------------------------
| •	feature parallel
| 一般的feature parallel就是对数据做垂直分割（partiion data vertically，就是对属性分割），然后将分割后的数据分散到各个worker上，各个workers计算其拥有的数据的best splits point, 之后再汇总得到全局最优分割点。但是lightgbm说这种方法通讯开销比较大，lightgbm的做法是每个worker都拥有所有数据，再分割？（没懂，既然每个worker都有所有数据了，再汇总有什么意义？这个并行体现在哪里？？）

| •	data parallel
| 传统的data parallel是将对数据集进行划分，也叫 平行分割(partion data horizontally)， 分散到各个workers上之后，workers对得到的数据做直方图，汇总各个workers的直方图得到全局的直方图。 lightgbm也claim这个操作的通讯开销较大，lightgbm的做法是使用”Reduce Scatter“机制，不汇总所有直方图，只汇总不同worker的不同feature的直方图(原理？)，在这个汇总的直方图上做split，最后同步。



有监督机器学习算法
========================

liner regression 线性回归
----------------------------------

lasso 回归和岭回归（ridge regression）其实就是在标准线性回归的基础上分别加入 L1 和 L2 正则化（regularization）

.. image:: ../../_static/machine_learning/lasso.png
	:align: center



岭回归和Lasso的区别
''''''''''''''''''''''''''''''''''
| Lasso是加 L1 penalty，也就是绝对值；岭回归是加 L2 penalty，也就是二范数。
| 从贝叶斯角度看，L1 正则项等价于参数 w 的先验概率分布满足拉普拉斯分布，而 L2 正则项等价于参数 w 的先验概率分布满足高斯分布。
| 从优化求解来看，岭回归可以使用梯度为零求出闭式解，而 Lasso 由于存在绝对值，在 0 处不可导，只能使用 Proximal Mapping 迭代求最优解。
| 从结果上看，L1 正则项会使得权重比较稀疏，即存在许多 0 值；L2 正则项会使权重比较小，即存在很多接近 0 的值。

| 从面经上拔下来的....有待考证


liner regression 矩阵解
''''''''''''''''''''''''''''''''''
.. image:: ../../_static/machine_learning/liner.png
	:align: center


Logistics regression
----------------------------
李宏毅视频


.. image:: ../../_static/machine_learning/lr.png
	:align: center
	
| 为什么 logistic regression 的输入特征一般是离散的而不是连续的？
| （1）离散特征的增加和减少都很容易，易于模型的快速迭代。 
| （2）稀疏向量内积乘法运算速度快，计算结果方便存储，容易扩展。 
| （3）对异常数据具有较强的鲁棒性。 
| （4）单个特征离散化为 N 个后，每个特征有单独的权重，相当于引入了非线性，增加了模型的表达能力，加大了拟合能力。 
| （5）可以特征交叉，M + N 个特征变为 M * N 个特征，进一步引入非线性，提升表达能力。 
| （6）特征离散化以后，起到了简化了逻辑回归模型的作用，降低了模型过拟合的风险。

用pytorch手写逻辑回归请见 leetcode那一页的非常规题 https://knowledge-record.readthedocs.io/zh-cn/latest/leetcode/leetcode.html#pytorch

逻辑回归的假设
''''''''''''''''''''''''''''''''''
数据服从伯努利分布 (伯努利分布：p 和 1-p)

模型的输出值是样本为正例的概率

为什么LR要用sigmoid
''''''''''''''''''''''''''''''''''
浅层：  值在0-1之间，连续，单调上升，光滑可导。关于0.5中心对称，符合LR要求预测值等于概率的要求。

深层：最大似然

正态分布解释

最大熵解释
该解释是说，在我们给定了某些假设之后，我们希望在给定假设前提下，分布尽可能的均匀。对于Logistic Regression，我们假设了对于{X,Y}，
我们预测的目标是Y|XY|X，并假设认为Y|XY|X服从bernoulli distribution，所以我们只需要知道P(Y|X)P(Y|X)；其次我们需要一个线性模型，所以P(Y|X)=f(wx)P(Y|X)=f(wx)。
接下来我们就只需要知道f是什么就行了。而我们可以通过最大熵原则推出的这个f，就是sigmoid


分类为什么用CE而不是MSE
''''''''''''''''''''''''''''''''''
| MSE作为分类的损失函数会有梯度消失的问题。
| MSE是非凸的，存在很多局部极小值点。

.. image:: ../../_static/machine_learning/cemse.png
	:align: center

非凸：

.. image:: ../../_static/machine_learning/cemse2.png
	:align: center

非凸应该是如果有很多个x，这些loss叠加起来是一个非凸的，因为是二次的叠加。



SVM
-------------
| https://www.bilibili.com/video/BV1ut41197F6?p=14
| 林轩田的 
| 包括李航的统计学习

SVM中的常考点以及手推SVM

机器学习--手推SVM以及KKT条件 https://zhuanlan.zhihu.com/p/45444502

手推SVM 支持向量机的简易推导和理解 https://blog.csdn.net/asd136912/article/details/79192239  这个讲的稍微简单些

.. image:: ../../_static/machine_learning/SVM.png
	:align: center
	
手推一下：

是一种二分类有监督算法，目标是最小间隔最大化，可以理解为一个求解凸二次规划问题

（函数间隔 、 几何间隔（对函数间隔做了归一化））

然后，使其满足KKT条件，变为二次凸优化问题，引入拉格朗日乘子

.. image:: ../../_static/machine_learning/SVM2.png
	:align: center

未完待续....


为什么要把原问题转化为对偶问题？
| （方便计算，方便引入核函数）
| 1.对偶问题将原始问题中的约束转为了对偶问题中的等式约束
| 2.方便核函数的引入
| 3.改变了问题的复杂度。由求特征向量w转化为求比例系数a，在原始问题下，求解的复杂度与样本的维度有关，即w的维度。在对偶问题下，只与样本数量有关。



.. image:: ../../_static/machine_learning/hinge_loss.png
	:align: center

为什么hinge loss在SVM时代大放异彩，但在神经网络时代就不好用了呢？主要就是因为svm时代我们用的是二分类，通过使用一些小技巧比如1 vs 1、1 vs n
等方式来做多分类问题。而如论文[3]这样直接把hinge loss应用在多分类上的话，当类别数特别大时，会有大量的非目标分数得到优化，
这样每次优化时的梯度幅度不等且非常巨大，极易梯度爆炸。

KNN
-------------------
.. image:: ../../_static/machine_learning/knn.png

样本的lable取决于与他最相近的k个样本的多数lable


朴素贝叶斯（Naive Bayes）
--------------------------------------------
李航统计学习
	
https://www.zhihu.com/question/19725590/answer/241988854

.. image:: ../../_static/machine_learning/bys.png
	:align: center

P(x | w1)这个x在w上的条件概率是有意义的，  因为可能存在P(y | w1)   （那个检测的问题   有患病、阳性、不患病、阴性）  luo

朴素贝叶斯的假设    "属性条件独立性假设   假设所有属性相互独立

我很喜欢这个解释：  链接：怎样用非数学语言讲解贝叶斯定理（Bayes's theorem）？ - 猴子的回答 - 知乎  https://www.zhihu.com/question/19725590/answer/241988854

.. image:: ../../_static/machine_learning/bys1.png
	:align: center

这里的P(A)是先验概率，P(B|A)/P(B)称为"可能性函数"（Likelyhood）。后验概率（新信息出现后的A概率）　＝　先验概率（A概率） * 可能性函数（新信息带来的调整）

| 如果"可能性函数"P(B|A)/P(B)>1，意味着"先验概率"被增强，事件A的发生的可能性变大；
| 如果"可能性函数"=1，意味着B事件无助于判断事件A的可能性；
| 如果"可能性函数"<1，意味着"先验概率"被削弱，事件A的可能性变小。



决策树
------------------------
| ID3 提出了初步的决策树算法，内部使用信息熵和信息增益来进行构建，每次迭代算则信息增益最大的特征属性作为分割属性。
| C4.5 提出了完整的决策树算法。使用信息增益率来取代ID3中的信息增益，在树的构造过程中会进行剪枝操作进行优化，能够自动完成对连续属性的离散化处理。
| CART (Classification And Regression Tree) 目前使用最多的决策树算法，选择那个使得划分后基尼指数最小的属性作为最优划分属性

| 一些资料
| https://www.jianshu.com/p/195d50a42ad5
|《李航 统计学习方法》 P60

信息增益
''''''''''''''''''''''''''''''''''
.. image:: ../../_static/machine_learning/熵.png
	:align: center
	:width: 500

.. image:: ../../_static/machine_learning/信息增益.png
	:align: center
	:width: 500

| 优点：
| 决策树构建速度快，实现简单。

| 缺点：
| 计算依赖于特征数目较多的特征，而属性值最多的属性并不一定最优。
| ID3算法不是递增算法。
| ID3算法是单变量决策树，对于特征属性之间的关系不会考虑。
| 抗噪性差。数据集中噪音点多可能会出现过拟合。
| 只适合小规模的数据集，需要将数据放到内存中。

信息增益率
''''''''''''''''''''''''''''''''''
.. image:: ../../_static/machine_learning/信息增益率.png
	:align: center
	:width: 500

g（D,A）是上面的的信息增益。g(D,A) = H(D) - H(D|A)

| 优点：
| 产生规则易于理解。
| 准确率较高。(因为考虑了连续值，数据越多拟合程度就越好。)
| 实现简单。

| 缺点：
| 对数据集需要进行多次扫描和排序，所以效率较低。(比如之前例子中收入的连续值，分割次数越多，需要扫描的次数也就越多，排序次数也越多。)
| 只适合小规模数据集，需要将数据放到内存中。

	
决策树的剪枝
''''''''''''''''''''''''''''''''''
.. image:: ../../_static/machine_learning/剪枝1.png
	:align: center
	:width: 500

设树的结点个数为|T|，则像正则化一样，损失函数加上 α|T|

基尼系数
''''''''''''''''''''''''''''''''''
.. image:: ../../_static/machine_learning/基尼系数1.png
	:align: center
	:width: 500
	
.. image:: ../../_static/machine_learning/基尼系数2.png
	:align: center
	:width: 500

cart算法使用基尼指数的主要目的：基尼指数的运算量比较小

分类树和回归树的区别
''''''''''''''''''''''''''''''''''
应用于分类和回归

分类树使用信息增益或增益比率来划分节点，回归树使用最大均方差划分节点

分类树：以C4.5分类树为例，穷举每一个feature的每一个阈值，找到使得按照feature<=阈值，和feature>阈值分成的两个分枝的熵最大的阈值，
按照该标准分枝得到两个新节点，用同样方法继续分枝直到所有人都被分入性别唯一的叶子节点，或达到预设的终止条件，若最终叶子节点中的性别不唯一，
则以多数人的性别作为该叶子节点的性别。

回归树：每个节点（不一定是叶子节点）都会得一个预测值，以年龄为例，该预测值等于属于这个节点的所有人年龄的平均值。
分枝时穷举每一个feature的每个阈值找最好的分割点，但衡量最好的标准不再是最大熵，而是最小化均方差即(每个人的年龄-预测年龄)^2 的总和 / N。
也就是被预测出错的人数越多，错的越离谱，均方差就越大，通过最小化均方差能够找到最可靠的分枝依据。分枝直到每个叶子节点上人的年龄都唯一或者
达到预设的终止条件(如叶子个数上限)，若最终叶子节点上人的年龄不唯一，则以该节点上所有人的平均年龄做为该叶子节点的预测年龄。


随机森林
--------------------------------------------

随机森林面试题

1.1 优缺点

| 优点。
| (1)不必担心过度拟合；
| (2)适用于数据集中存在大量未知特征；
| (3)能够估计哪个特征在分类中更重要；
| (4)具有很好的抗噪声能力；
| (5)算法容易理解；
| (6)可以并行处理。

| 缺点。
| （1）对小量数据集和低维数据集的分类不一定可以得到很好的效果。
| （2）执行速度虽然比Boosting等快，但是比单个的决策树慢很多。
| （3）可能会出现一些差异度非常小的树，淹没了一些正确的决策。
| （4）由于树是随机生成的，结果不稳定（kpi值比较大）

| 1.2 生成步骤介绍
| 1、从原始训练数据集中，应用bootstrap方法有放回地随机抽取k个新的自助样本集，并由此构建k棵分类回归树，每次未被抽到的样本组成了Ｋ个袋外数据（out-of-bag,BBB）。
| 2、设有n 个特征，则在每一棵树的每个节点处随机抽取mtry 个特征，通过计算每个特征蕴含的信息量，特征中选择一个最具有分类能力的特征进行节点分裂。
| 3、每棵树最大限度地生长， 不做任何剪裁
| 4、将生成的多棵树组成随机森林， 用随机森林对新的数据进行分类， 分类结果按树分类器投票多少而定。

| 1.3 随机森林与SVM的比较
| （1）不需要调节过多的参数，因为随机森林只需要调节树的数量，而且树的数量一般是越多越好，而其他机器学习算法，比如SVM，有非常多超参数需要调整，如选择最合适的核函数，正则惩罚等。
| （2）分类较为简单、直接。随机森林和支持向量机都是非参数模型（复杂度随着训练模型样本的增加而增大）。相较于一般线性模型，就计算消耗来看，训练非参数模型因此更为耗时耗力。分类树越多，需要更耗时来构建随机森林模型。同样，我们训练出来的支持向量机有很多支持向量，最坏情况为，我们训练集有多少实例，就有多少支持向量。虽然，我们可以使用多类支持向量机，但传统多类分类问题的执行一般是one-vs-all（所谓one-vs-all 就是将binary分类的方法应用到多类分类中。比如我想分成K类，那么就将其中一类作为positive），因此我们还是需要为每个类训练一个支持向量机。相反，决策树与随机深林则可以毫无压力解决多类问题。
| （3）比较容易入手实践。随机森林在训练模型上要更为简单。你很容易可以得到一个又好且具鲁棒性的模型。随机森林模型的复杂度与训练样本和树成正比。支持向量机则需要我们在调参方面做些工作，除此之外，计算成本会随着类增加呈线性增长。
| （4）小数据上，SVM优异，而随机森林对数据需求较大。就经验来说，我更愿意认为支持向量机在存在较少极值的小数据集上具有优势。随机森林则需要更多数据但一般可以得到非常好的且具有鲁棒性的模型。

| 1.4 随机森林不会发生过拟合的原因
| 在建立每一棵决策树的过程中，有两点需要注意-采样与完全分裂。首先是两个随机采样的过程，random forest对输入的数据要进行行、列的采样。对于行采样，采用有放回的方式，也就是在采样得到的样本集合中，可能有重复的样本。
| 对于行采样，采用有放回的方式，也就是在采样得到的样本集合中，可能有重复的样本。假设输入样本为N个，那么采样的样本也为N个。这样使得在训练的时候，每一棵树的输入样本都不是全部的样本，使得相对不容易出现over-fitting。*然后进行列采样，从M 个feature中，选择m个(m << M)。之后就是对采样之后的数据使用完全分裂的方式建立出决策树，这样决策树的某一个叶子节点要么是无法继续分裂的，要么里面的所有样本的都是指向的同一 个分类。*一般很多的决策树算法都一个重要的步骤 - 剪枝，但是这里不这样干，由于之前的两个随机采样的过程保证了随机性，所以就算不剪枝，也不会出现over-fitting。

| 1.5 随机森林与梯度提升树（GBDT）区别
| 随机森林：决策树+bagging=随机森林
| 梯度提升树：决策树+Boosting=GBDT
| 两者区别在于bagging boosting之间的区别。
| 像神经网络这样为消耗时间的算法，bagging可通过并行节省大量的时间开销
| baging和boosting都可以有效地提高分类的准确性
| baging和boosting都可以有效地提高分类的准确性
| 一些模型中会造成模型的退化（过拟合）
| boosting思想的一种改进型adaboost方法在邮件过滤，文本分类中有很好的性能。


随机森林随机性
''''''''''''''''''''''''''''''''''
随机森林的随机性体现在每颗树的训练样本是随机的，树中每个节点的分裂属性集合也是随机选择确定的。

随机森林需要剪枝吗
''''''''''''''''''''''''''''''''''
不需要，后剪枝是为了避免过拟合，随机森林随机选择变量与树的数量，已经避免了过拟合，没必要去剪枝了。

为什么要有放回的抽样
''''''''''''''''''''''''''''''''''
保证样本集间有重叠，若不放回，每个训练样本集及其分布都不一样，可能导致训练的各决策树差异性很大，最终多数表决无法 “求同”，即最终多数表决相当于“求同”过程。

影响性能因素
''''''''''''''''''''''''''''''''''
| •单棵树的分类强度：每棵树分类强度越大，随机森林分类性能越好
| •森林中树之间的相关度：树之间的相关度越大，则随机森林的分类性能越差

聚类
==============================

资料
------------------
清华大学【数据挖掘：聚类分析】  https://www.bilibili.com/video/BV1Vt411v7YS?p=1

机器学习中的聚类算法演变及学习笔记  https://www.nowcoder.com/discuss/432266?type=post&order=create&pos=&page=0&channel=666&source_id=search_post

聚类的种类
--------------------------
| 基于划分的聚类
| K-Means

| 基于密度的聚类
| Mean-Shift
| DBSCAN

| 基于概率模型的聚类
| 高斯混合模型（GMM）的最大期望（EM）


| 基于层次的聚类
| AGNES
| BIRCH

其他方法

Kmeans
------------------------
.. image:: ../../_static/machine_learning/kmeans.png
	:align: center
	:width: 500


K-Means聚类的优点：
''''''''''''''''''''''''''''''''''
| •	原理简单，实现容易，收敛速度快
| •	参数只有K，计算复杂度相对较低
| •	模型可解释性强


K-Means聚类的缺点：
''''''''''''''''''''''''''''''''''
| •	需要事先确定聚类的簇数（即K值）
| •	对簇中心初始值的选取比较敏感
| •	对噪声和离群点很敏感
| •	采用迭代方法，容易陷入局部最优
| •	适用场景有限，不能解决非凸数据



K值的选取
''''''''''''''''''''''''''''''''''
| •	根据数据的可视化分布情况，结合对业务场景理解，人工选定K值
| •	Elbow method（即手肘法则,类似PCA的降维选维度）：通过WSS随聚类数量K的变化曲线，取手肘位置的K（例如Gap Statistic、Jump Statistic等）
| •	通过计算类内内聚程度和类间分离度来确定K（例如使用平均轮廓系数、类内距离/类间距离等）
| •	其他：例如使用ISODATA、Gap Statistic公式、计算不同K值下的BIC/AIC、X-means clustering（AIC/BIC）等



K-Means聚类变体
''''''''''''''''''''''''''''''''''
| •	k-means++

| 考虑到K-Means对簇中心初始值的选取比较敏感，同类的还有：intelligent k-means、genetic k-means、CLARANS等。

| 在选取第一个聚类中心(n=1)时，同样是通过随机的方法。
| 在选取第n+1个聚类中心时，距离当前n个聚类中心越远的点会有更高的概率被选为第n+1个聚类中心。

| •	k-medians

| 考虑到k-means对噪声和离群值很敏感，同类的还有k-medoids

| k-medians对于中心点的选取方式是中位值。原因在于，噪声和离群点对中位值的变化影响不大。但是需要排序，速度较慢。

| •	k-modes

| 考虑到k-means不适用于类别型数据

| k-modes算法采用差异度来代替k-means算法中的距离。k-modes算法中差异度越小，则表示距离越小。

| •	kernel k-means

| 考虑到k-means不能解决非凸数据，同类的还有谱聚类等。

| kernel k-means通过一个非线性映射，将输入空间中的数据点映射到一个高维特征空间中，使得样本在核空间线性可分，在特征空间聚类。
| 值得一提的是，谱聚类算法是建立在图论中的谱图理论基础上，其本质是将聚类问题转化为图的最优划分问题，是一种点对聚类算法。



GMM EM
-----------------
目前的理解是： 
kmeans是先随机初始化一些中心点，然后根据距离重新划分数据集，然后选择新的中心点，再重新划分数据集   

那GMM这里，看起来是首先随机选取几个高斯分布，然后分布计算每个点属于某个高斯分布的概率

看起来像是把kmeans用距离划分改成了 用 高斯分布的概率 ？

DBSCAN
--------------------

20聚类算法-DBSCAN  https://www.bilibili.com/video/BV1j4411H7xv?p=1

核心思想....类似传销，发展下线直到不能发展为止

.. image:: ../../_static/machine_learning/DBSCAN1.png
	:align: center

核心点就是划分一个半径，圆内被圈到的数据数量要求大于阈值

.. image:: ../../_static/machine_learning/DBSCAN2.png
	:align: center
	
.. image:: ../../_static/machine_learning/DBSCAN3.png
	:align: center
	
	
不能被发展成下线又不能自成一体的就是离群点。

流程：

.. image:: ../../_static/machine_learning/DBSCAN4.png
	:align: center

DBSCAN的主要优点有：

1）可以对任意形状的稠密数据集进行聚类，相对的，K-Means之类的聚类算法一般只适用于凸数据集。

2）可以在聚类的同时发现异常点，对数据集中的异常点不敏感。

3）聚类结果没有偏倚，相对的，K-Means之类的聚类算法初始值对聚类结果有很大影响。

DBSCAN的主要缺点有：

1）如果样本集的密度不均匀、聚类间距差相差很大时，聚类质量较差，这时用DBSCAN聚类一般不适合。

2）如果样本集较大时，聚类收敛时间较长，此时可以对搜索最近邻时建立的KD树或者球树进行规模限制来改进。

3）调参相对于传统的K-Means之类的聚类算法稍复杂，主要需要对距离阈值ϵ，邻域样本数阈值MinPts联合调参，不同的参数组合对最后的聚类效果有较大影响。



kmeans 球形 而且倾向于簇的形状一样大
GMM 高斯分布球形  
DBSCAN 不要求形状一样

AGNES聚类
------------------
.. image:: ../../_static/machine_learning/AGNES.png
	:align: center


AGNES聚类的优点：

| 距离和规则的相似度容易定义，限制少
| 不需要预先制定聚类数
| 可以发现类的层次关系
| 可以聚类成其它形状

AGNES聚类的缺点：

| 计算复杂度太高
| 奇异值也能产生很大影响
| 算法很可能聚类成链状

sequential leader clustering
----------------------------------

.. image:: ../../_static/machine_learning/sequential-leader-clustering.png
	:align: center

聚类的衡量
--------------------
类内距离和类间距离

？？？


其他常见问题
======================

如何解决机器学习中样本不均衡问题？
------------------------------------------
| •	通过过抽样和欠抽样解决样本不均衡

| 抽样是解决样本分布不均衡相对简单且常用的方法，包括过抽样和欠抽样两种。

| 过抽样
| 过抽样（也叫上采样、over-sampling）方法通过增加分类中少数类样本的数量来实现样本均衡，最直接的方法是简单复制少数类样本形成多条记录，这种方法的缺点是如果样本特征少而可能导致过拟合的问题；经过改进的过抽样方法通过在少数类中加入随机噪声、干扰数据或通过一定规则产生新的合成样本，例如SMOTE算法。

| 欠抽样
| 欠抽样（也叫下采样、under-sampling）方法通过减少分类中多数类样本的样本数量来实现样本均衡，最直接的方法是随机地去掉一些多数类样本来减小多数类的规模，缺点是会丢失多数类样本中的一些重要信息。

| 总体上，过抽样和欠抽样更适合大数据分布不均衡的情况，尤其是第一种（过抽样）方法应用更加广泛。

| •	通过正负样本的惩罚权重解决样本不均衡

| 通过正负样本的惩罚权重解决样本不均衡的问题的思想是在算法实现过程中，对于分类中不同样本数量的类别分别赋予不同的权重（一般思路分类中的小样本量类别权重高，大样本量类别权重低），然后进行计算和建模。
| 使用这种方法时需要对样本本身做额外处理，只需在算法模型的参数中进行相应设置即可。很多模型和算法中都有基于类别参数的调整设置，以scikit-learn中的SVM为例，通过在class_weight
| : {dict, 'balanced'}中针对不同类别针对不同的权重，来手动指定不同类别的权重。如果使用其默认的方法balanced，那么SVM会将权重设置为与不同类别样本数量呈反比的权重来做自动均衡处理，计算公式为：n_samples / (n_classes * np.bincount(y))。
| 如果算法本身支持，这种思路是更加简单且高效的方法。

| •	通过组合/集成方法解决样本不均衡
| 组合/集成方法指的是在每次生成训练集时使用所有分类中的小样本量，同时从分类中的大样本量中随机抽取数据来与小样本量合并构成训练集，这样反复多次会得到很多训练集和训练模型。最后在应用时，使用组合方法（例如投票、加权投票等）产生分类预测结果。
| 例如，在数据集中的正、负例的样本分别为100和10000条，比例为1:100。此时可以将负例样本（类别中的大量样本集）随机分为100份（当然也可以分更多），每份100条数据；然后每次形成训练集时使用所有的正样本（100条）和随机抽取的负样本（100条）形成新的数据集。如此反复可以得到100个训练集和对应的训练模型。
| 这种解决问题的思路类似于随机森林。在随机森林中，虽然每个小决策树的分类能力很弱，但是通过大量的“小树”组合形成的“森林”具有良好的模型预测能力。
| 如果计算资源充足，并且对于模型的时效性要求不高的话，这种方法比较合适。

| •	通过特征选择解决样本不均衡
| 上述几种方法都是基于数据行的操作，通过多种途径来使得不同类别的样本数据行记录均衡。除此以外，还可以考虑使用或辅助于基于列的特征选择方法。
| 一般情况下，样本不均衡也会导致特征分布不均衡，但如果小类别样本量具有一定的规模，那么意味着其特征值的分布较为均匀，可通过选择具有显著型的特征配合参与解决样本不均衡问题，也能在一定程度上提高模型效果。
| 提示 上述几种方法的思路都是基于分类问题解决的。实际上，这种从大规模数据中寻找罕见数据的情况，也可以使用非监督式的学习方法，例如使用One-class SVM进行异常检测。分类是监督式方法，前期是基于带有标签（Label）的数据进行分类预测；而采用非监督式方法，则是使用除了标签以外的其他特征进行模型拟合，这样也能得到异常数据记录。所以，要解决异常检测类的问题，先是考虑整体思路，然后再考虑方法模型。


数据挖掘中常见的「异常检测」算法有哪些？
------------------------------------------------
| https://www.zhihu.com/question/280696035
| 1. 无监督异常检测

| 如果归类的话，无监督异常检测模型可以大致分为：

| •	统计与概率模型（statistical and probabilistic and models）：主要是对数据的分布做出假设，并找出假设下所定义的“异常”，因此往往会使用极值分析或者假设检验。比如对最简单的一维数据假设高斯分布，然后将距离均值特定范围以外的数据当做异常点。而推广到高维后，可以假设每个维度各自独立，并将各个维度上的异常度相加。如果考虑特征间的相关性，也可以用马氏距离（mahalanobis distance）来衡量数据的异常度[12]。不难看出，这类方法最大的好处就是速度一般比较快，但因为存在比较强的“假设”，效果不一定很好。

| •	线性模型（linear models）：假设数据在低维空间上有嵌入，那么无法、或者在低维空间投射后表现不好的数据可以认为是离群点。举个简单的例子，PCA可以用于做异常检测[10]，一种方法就是找到k个特征向量（eigenvector），并计算每个样本再经过这k个特征向量投射后的重建误差（reconstruction error），而正常点的重建误差应该小于异常点。同理，也可以计算每个样本到这k个选特征向量所构成的超空间的加权欧氏距离（特征值越小权重越大）。在相似的思路下，我们也可以直接对协方差矩阵进行分析，并把样本的马氏距离（在考虑特征间关系时样本到分布中心的距离）作为样本的异常度，而这种方法也可以被理解为一种软性（Soft PCA） [6]。同时，另一种经典算法One-class SVM[3]也一般被归类为线性模型。

| •	基于相似度衡量的模型（proximity based models）：异常点因为和正常点的分布不同，因此相似度较低，由此衍生了一系列算法通过相似度来识别异常点。比如最简单的K近邻就可以做异常检测，一个样本和它第k个近邻的距离就可以被当做是异常值，显然异常点的k近邻距离更大。同理，基于密度分析如LOF [1]、LOCI和LoOP主要是通过局部的数据密度来检测异常。显然，异常点所在空间的数据点少，密度低。相似的是，Isolation Forest[2]通过划分超平面来计算“孤立”一个样本所需的超平面数量（可以想象成在想吃蛋糕上的樱桃所需的最少刀数）。在密度低的空间里（异常点所在空间中），孤例一个样本所需要的划分次数更少。另一种相似的算法ABOD[7]是计算每个样本与所有其他样本对所形成的夹角的方差，异常点因为远离正常点，因此方差变化小。换句话说，大部分异常检测算法都可以被认为是一种估计相似度，无论是通过密度、距离、夹角或是划分超平面。通过聚类也可以被理解为一种相似度度量，比较常见不再赘述。

| •	集成异常检测与模型融合：在无监督学习时，提高模型的鲁棒性很重要，因此集成学习就大有用武之地。比如上面提到的Isolation Forest，就是基于构建多棵决策树实现的。最早的集成检测框架feature bagging[9]与分类问题中的随机森林（random forest）很像，先将训练数据随机划分（每次选取所有样本的d/2-d个特征，d代表特征数），得到多个子训练集，再在每个训练集上训练一个独立的模型（默认为LOF）并最终合并所有的模型结果（如通过平均）。值得注意的是，因为没有标签，异常检测往往是通过bagging和feature bagging比较多，而boosting比较少见。boosting情况下的异常检测，一般需要生成伪标签，可参靠[13, 14]。集成异常检测是一个新兴但很有趣的领域，综述文章可以参考[16, 17, 18]。

| •	特定领域上的异常检测：比如图像异常检测 [21]，顺序及流数据异常检测（时间序列异常检测）[22]，以及高维空间上的异常检测 [23]，比如前文提到的Isolation Forest就很适合高维数据上的异常检测。


| 维度低的时候，二维 可以直接用高斯函数的3希格玛原则，低维，KNN，实际上是计算相似度，再高维的话可以isolation Forrest， 之后两个月我可以学一下  （pca或者autoencoder降维 再高斯）

sklearn

https://scikit-learn.org/stable/modules/outlier_detection.html#overview-of-outlier-detection-methods


上采用 & 下采样
---------------------
https://www.cnblogs.com/zhanjiahui/p/11643544.html

https://www.jianshu.com/p/fd9e2166cfcc


几种距离度量方法比较
-----------------------
https://blog.csdn.net/J_Boom/article/details/86763024


欧氏距离

.. image:: ../../_static/machine_learning/欧氏距离.png
	:align: center
	:width: 400

曼哈顿距离

.. image:: ../../_static/machine_learning/曼哈顿距离.png
	:align: center
	:width: 400

切比雪夫距离

.. image:: ../../_static/machine_learning/切比雪夫距离.png
	:align: center
	:width: 400

| 马氏距离
| 就是做个PCA 排除均值和方差的影响

.. image:: ../../_static/machine_learning/马氏距离.png
	:align: center
	:width: 400

余弦距离 略

汉明距离(Hamming Distance)  就是编辑距离

杰卡德距离(Jaccard Distance)

.. image:: ../../_static/machine_learning/杰卡德距离.png
	:align: center

相关距离(Correlation distance)

.. image:: ../../_static/machine_learning/相关距离.png
	:align: center


周期性特征的编码问题
----------------------------------
对于周期性的变量，如日期，月，日，时，分，单纯用数值表示或者简单按数值可取数量编码是欠妥的，如23时和凌晨1h，二者相差只有2h，但是如果只是将时按简单的数字做特征，23与1，二者相差22h，将严重误导模型学习的结果。所以有必要对诸如小时，分钟这样的周期性特征做合适的编码工作。最典型的编码方式是将一维数值变量扩展为二维的（正弦值，余弦值）来编码。步骤如下：

  1.某特征X，计算其最大取值max_value，如小时的最大取值是23时，max_value = 23

  2.计算正弦值余弦值：

	.. image:: ../../_static/machine_learning/circle_encode4.png
		:align: left


  3.将扩充后的特征Xsin，Xcos加入到特征集合中，去除其对应的原特征X（不用单独的“时”数值特征，用“时”的sin，cos值代替）

具体说一下

计算sin，cos的方式，就是普通的数值转弧度制，计算正弦余弦值。

而单单把一维变量转化为同样是一维的sin/cos的话，由于sin/cos的周期性，会带来同一取值对应多个不同时刻的问题，如下图

第一张sin，第二张cos，数据X：0,1,2,3,...,23，Y：对应的sin(x),cos(x)

.. image:: ../../_static/machine_learning/circle_encode1.png
	:width: 300

.. image:: ../../_static/machine_learning/circle_encode2.png
	:width: 300


单独的一维sin/cos并不能限制取值的唯一性，而sin,cos组合便可以达到这个目的


.. image:: ../../_static/machine_learning/circle_encode3.png
	:width: 300



启发式算法
-----------------------------
通俗的解释就是利用类似仿生学的原理，将自然、动物中的一些现象抽象成为算法处理相应问题。当一个问题是NP难问题时，是无法求解到最优解的，
因此，用一种相对好的求解算法，去尽可能逼近最优解，得到一个相对优解，在很多实际情况中也是可以接受的。

举例：模拟退火算法（SA）、遗传算法（GA）、蚁群算法（ACO）、人工神经网络（ANN）



生成式和判别式 算法
----------------------------
.. image:: ../../_static/machine_learning/scpb.png
	:align: center
	:width: 400

机器学习“判定模型”和“生成模型”有什么区别？ https://www.zhihu.com/question/20446337/answer/256466823


举一个例子：判别式模型举例：要确定一个羊是山羊还是绵羊，用判别模型的方法是从历史数据中学习到模型，
然后通过提取这只羊的特征来预测出这只羊是山羊的概率，是绵羊的概率。

生成式模型举例：利用生成模型是根据山羊的特征首先学习出一个山羊的模型，
然后根据绵羊的特征学习出一个绵羊的模型，然后从这只羊中提取特征，放到山羊模型中看概率是多少，在放到绵羊模型中看概率是多少，哪个大就是哪个。


细细品味上面的例子，判别式模型是根据一只羊的特征可以直接给出这只羊的概率（比如logistic regression，这概率大于0.5时则为正例，否则为反例），
而生成式模型是要都试一试，最大的概率的那个就是最后结果

在机器学习中任务是从属性X预测标记Y，判别模型求的是P(Y|X)，即后验概率；
而生成模型最后求的是P(X,Y)，即联合概率。从本质上来说：判别模型之所以称为“判别”模型，是因为其根据X“判别”Y；而生成模型之所以称为“生成”模型，
是因为其预测的根据是联合概率P(X,Y)，而联合概率可以理解为“生成”(X,Y)样本的概率分布（或称为 依据）；具体来说，机器学习已知X，从Y的候选集合中选出一个来，
可能的样本有(X,Y_1), (X,Y_2), (X,Y_3),……，(X,Y_n),实际数据是如何“生成”的依赖于P(X,Y)，那么最后的预测结果选哪一个Y呢？那就选“生成”概率最大的那个吧~

.. image:: ../../_static/machine_learning/生成式判别式.png
	:align: center


L0 L1 L2 正则化
-------------------
| L0正则化的值是模型参数中非零参数的个数。
| L1正则化表示各个参数绝对值之和。
| L2正则化标识各个参数的平方的和的开方值。

| L1 和 L2：
| •	L2正则相比于L1正则来说，得到的解比较平滑（不是稀疏），但是同样能够保证解中接近于0（但不是等于0，所以相对平滑）的维度比较多，降低模型的复杂度。
| •	L2 计算起来更方便，而 L1 在特别是非稀疏向量上的计算效率就很低；
| •	L1 最重要的一个特点，输出稀疏，会把不重要的特征直接置零，而 L2 则不会；
| •	L2 有唯一解，而 L1 不是。


两种正则化会导致模型最后有什么不同，为什么会有这种现象

L1 和 L2 正则的区别是什么
''''''''''''''''''''''''''''''''''

李飞飞在CS2312中给的更为详细的解释：

L2正则化可以直观理解为它对于大数值的权重向量进行严厉惩罚，倾向于更加分散的权重向量。由于输入和权重之间的乘法操作，这样就有了一个优良的特性：
使网络更倾向于使用所有输入特征，而不是严重依赖输入特征中某些小部分特征。 这样做可以提高模型的泛化能力，降低过拟合的风险。

L1正则化会让权重向量在最优化的过程中变得稀疏（即非常接近0）。也就是说，使用L1正则化的神经元最后使用的是它们最重要的输入数据的稀疏子集，同时对于噪音输入则几乎是不变的了。

在实践中，如果不是特别关注某些明确的特征选择，一般说来L2正则化都会比L1正则化效果好。



这个问题可以从两个角度去解释，概率角度和微积分角度。

首先是概率角度。
正则项来自于对数据的先验知识，这个先验知识的概率密度函数定义为 p(x)。如果我们认为，数据是服从高斯分布的，那么就应该在代价函数中加入数据先验P(x),
一般由于推导和计算方便会加入对数似然,也就是log(P(x)),然后再去优化,如果你去看看高斯分布的概率密度函数P(x),你会发现取对数后的log(P(x))就剩下一个平方项了,这就是L2范式的由来--高斯先验.

同样,如果你认为你的数据是稀疏的,不妨就认为它来自某种laplace分布.不知你是否见过laplace分布的概率密度函数，laplace分布是尖尖的分布,
是不是很像一个pulse?从这张图上,你应该就能看出,服从laplace分布的数据就是稀疏的了，如果取对数,剩下的是一个一次项|x-u|,这就是L1范式.
所以用L1范式去正则,就假定了你的数据是laplace分布,是稀疏的.

微积分角度。

一个优化问题的最优解，一般是在导数 = 0 的位置上。

如果原有模型的参数不是稀疏的，那么就意味着损失函数 f(x) 在求导时，0 点的导数不等于 0 ，即 f'(0) != 0，否则 如果等于 0 的话，那么 0 会是一个局部解导致模型稀疏。

此时，如果加上一个 L2 正则项，原有的 损失函数就变成了 f(x) + C||x||^2， 它在 0 点的导数就是 f'(0) + 2Cx (x = 0)。 因为 f'(0) != 0 所以整个式子不等于 0 ，所以 x = 0 不是极值点。

如果不是 L2， 是 L1，那么 损失函数就变成了 f(x) + C|x|，其 0 点 左导数 -C+f'(0), 右导数是 C+f'(0) ， 从而当C > |f'(0)|的时候，次梯度集合是包含0点的，
而根据次梯度的定义，这个时候 x=0 即为最小值。


Normalization & Standardization
------------------------------------------------------
Normalization typically means rescaling the values into a range of [0,1].

Standardization typically means rescaling values to have a mean of 0 and a standard deviation of 1.

记忆： Standardization 就是 **standard deviation** of 1 and mean of 0

PCA
---------------
单层线性神经网络的降维=PCA ？？

核心思想，我的总结：在高维空间中散布着很多点，要找到一条特征向量Eigenvector  第一主成分，使得这些点在投影到这个特征向量上以后是分散的最开的。如何衡量分散程度？
用Var 方差最大来衡量。


数学推导的话：PCA（主成分分析法）中，主成分方向的推导 https://www.bilibili.com/video/BV1ED4y1U7CC?from=search&seid=17109712823241897967
这个真的讲得好！颇像当时在JHU上课时学的讲法

关于点积： 求一个点在一个向量上的投影，就用点积。

所以说Var[aTX]要最大。a是那个特征向量

For any vector  a∈RN 

𝕍𝕒𝕣[aTX]=𝔼[(aTX)(XTa)]=𝔼[aT(XXT)a]Var[aTX]=E[(aTX)(XTa)]=E[aT(XXT)a] 

so

𝕍𝕒𝕣[aTX]=aT𝔼[XXT]a=aTCaVar[aTX]=aTE[XXT]a=aT*C*a  其中C=XT*X 是一个实对称矩阵

We have to maximize this such that  a**2=1也就是aT*a=1,做个单位化  （不然的话，让aT*C*a大，只需要让a越来越大就好） 注意，这里已经做了μ=0的平移变换了

这是一个优化问题，有对a的限制，用拉格朗日乘子法，转化为求 u(a)=aT*C*a - λ(aT*a-1)的最大值

这个是矩阵的求导有点复杂。可以简单的看成 Ca**2-λa**2。求导的话，是求Ca-λa=0 ===> (C-λI)a=0

当 a,λ 分别为C矩阵的特征向量，特征值时，u(a)有极值

这样一来，可以直接求解C的特征向量和特征值，将特征值从大到小排序，所对应的特征向量作为PCA的轴。

关于如何通过一个给定的矩阵求解他的特征向量和特征值，手算的话请看https://blog.csdn.net/Junerror/article/details/80222540

.. image:: ../../_static/machine_learning/特征值的求解.png
	:align: center


JHU上课时画的那个图，长得像loss下降的形式是这个意思。比如说前几个最大的λ的值是10,6,1。那么从三维降维成两维,保留的信息就是(10+6)/(10+6+1)

LDA(Linear Discriminant Analysis)  线性判别分析
--------------------------------------------------

LDA是一种**监督学习**的降维技术，也就是说它的数据集的每个样本是有类别输出的。这一点和PCA不一样，PCA是**无监督学习**

LDA的基本思想：给定训练样例集，设法将样例投影到一条直线上，使得同类样例的投影点尽可能接近、异类样例的投影点中心尽可能远离。更简单的概括为一句话，就是“投影后类内方差最小，类间方差最大”。

.. image:: ../../_static/machine_learning/LDA1.png
	:align: center
	
周志华《机器学习》

核心思想，我的总结：在高维空间中散布着很多点，已知label。要找到一条特征向量Eigenvector，使得这些点在投影到这个特征向量上以后，同一标签的数据间隔最小，不同标签的数据间隔最大。
如何衡量分散程度？用Var 方差最大来衡量。


LDA(Latent Dirichlet Allocation)  隐含狄利克雷分布
------------------------------------------------------------------------

常常用于浅层语义分析，在文本语义分析中是一个很有用的模型。

LDA模型是一种主题模型，它可以将文档集中的每篇文档的主题以概率分布的形式给出，从而通过分析一些文档抽取出它们的主题（分布）出来后，便可以根据主题（分布）进行主题聚类或文本分类。

同时，它是一种典型的词袋模型，即一篇文档是由一组词构成，词与词之间没有先后顺序的关系。

LDA模型就是要根据给定一篇文档，推断这个文档的主题是什么，并给出各个主题的概率大小是多少。



参数稀疏有什么好处
------------------------------
1）特征选择(Feature Selection)： 大家对稀疏规则化趋之若鹜的一个关键原因在于它能实现特征的自动选择。一般来说，xi的大部分元素（也就是特征）
都是和最终的输出yi没有关系或者不提供任何信息的，在最小化目标函数的时候考虑xi这些额外的特征，虽然可以获得更小的训练误差，但在预测新的样本时，
这些没用的信息反而会被考虑，从而干扰了对正确yi的预测。稀疏规则化算子的引入就是为了完成特征自动选择的光荣使命，它会学习地去掉这些没有信息的特征，也就是把这些特征对应的权重置为0。

2）可解释性(Interpretability)： 另一个青睐于稀疏的理由是，模型更容易解释。例如患某种病的概率是y，然后我们收集到的数据x是1000维的，
也就是我们需要寻找这1000种因素到底是怎么影响患上这种病的概率的。假设我们这个是个回归模型：y=w1*x1+w2*x2+…+w1000*x1000+b
（当然了，为了让y限定在[0,1]的范围，一般还得加个Logistic函数）。通过学习，如果最后学习到的w*就只有很少的非零元素，例如只有5个非零的wi，
那么我们就有理由相信，这些对应的特征在患病分析上面提供的信息是巨大的，决策性的。也就是说，患不患这种病只和这5个因素有关，那医生就好分析多了。
但如果1000个wi都非0，医生面对这1000种因素.

Rank Averaging
-----------------------------
.. image:: ../../_static/machine_learning/Rank_Averaging.png
	:align: center



有哪些常见的 Feature engineering 特征工程的方法
----------------------------------------------------
常见的特征工程包括：

异常值处理 Outlier Handling
''''''''''''''''''''''''''''''''''
删除异常值 Outlier Removal，长尾截断 Long-tail Truncation，缺失值处理 Missing Value Handling

对于缺失值，首先要做数据统计，看看是不是上游数据出现了问题，缺失值多不多，占比多少，和过去相比怎么样

| 如果正常，可以用
| 删除缺失值：删除包含缺失值的样本或特征。
| 填补缺失值：用均值、中位数、众数或其他值填补缺失值。
| 插值法：利用插值方法填补缺失值，如线性插值或多项式插值。

数值特征处理、类别特征处理 Numerical and Categorical Feature Engineering
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
数值特征处理：

| 归一化（Normalization）：将数值特征缩放到一个特定的范围（如[0, 1]）。
| 标准化（Standardization）：将数值特征调整为均值为0，标准差为1的分布。
| 分桶（Binning）：将连续型变量分成多个区间（如将年龄分为“青年”、“中年”、“老年”）。
| 多项式特征（Polynomial Features）：生成特征的多项式组合（如x, x², x³）。


类别特征处理：

| 独热编码（One-Hot Encoding）：将类别特征转换为二进制向量。
| 标签编码（Label Encoding）：将类别特征转换为整数值。
| 目标编码（Target Encoding）：用目标变量的统计值（如平均值）来替换类别特征。


特征构造 Feature Construction
''''''''''''''''''''''''''''''''''
组合特征：比如相加，相乘等等 Feature Combination (e.g., addition, multiplication, etc.)


降维 Dimensionality Reduction
''''''''''''''''''''''''''''''''''
PCA LDA SVD Autoencoder

特征过多/维度灾难/解决方案 https://knowledge-record.readthedocs.io/zh-cn/latest/machine_learning/machine_learning.html#id46

维度过高会导致样本在特征空间中分布稀疏



数据清洗
-----------------------
数据清洗一是为了解决数据质量问题，二是让数据更适合做挖掘。

解决数据质量问题
''''''''''''''''''''''''''''''''''
| 1.	数据的完整性----例如人的属性中缺少性别、籍贯、年龄等
| 2.	数据的唯一性----例如不同来源的数据出现重复的情况
| 3.	数据的权威性----例如同一个指标出现多个来源的数据，且数值不一样
| 4.	数据的合法性----例如获取的数据与常识不符，年龄大于150岁
| 5.	数据的一致性----例如不同来源的不同指标，实际内涵是一样的，或是同一指标内涵不一致

让数据更适合做挖掘或展示
''''''''''''''''''''''''''''''''''
| 1.	高维度----不适合挖掘
| 2.	维度太低----不适合挖掘
| 3.	无关信息----减少存储
| 4.	字段冗余----一个字段是其他字段计算出来的，会造成相关系数为1或者主成因分析异常）
| 5.	多指标数值、单位不同----如GDP与城镇居民人均收入数值相差过大



特征过多/维度灾难/解决方案
----------------------------------------
维度灾难：https://zhuanlan.zhihu.com/p/26945814

样本在特征空间中分布稀疏

使用太多特征导致过拟合。分类器学习了过多样本数据的异常特征(噪声)，而对新数据的泛化能力不好。

解决方案
''''''''''''''''''''''''''''''''''
1.L1正则化（Lasso）:

| L1正则化（Lasso）可以推动模型将一些特征的权重归零，从而实现特征选择。

2.主成分分析 (Principal Component Analysis, PCA):

| PCA是一种降维技术，可以将高维数据映射到低维空间，保留最重要的特征。通过保留主成分，可以减少数据的维度，同时尽量保留原始数据的方差。

3.t-Distributed Stochastic Neighbor Embedding  （t-SNE）

| 是一种非线性降维方法，可以在可视化和特征提取中使用。主要用于可视化，特别是高维数据降维到2维或者3维

4.递归特征消除 (Recursive Feature Elimination, RFE):

| RFE是一种递归的特征选择方法，获取每个特征的重要程度，剔除最不重要的特征，不断的重复递归这个步骤，直到达到所需的特征数量。常用于对模型的复杂度进行控制，同时选择最相关的特征。

5.稳定性选择 (Stability Selection):

| 通过在不同的子样本上运行特征选择算法，可以提高特征选择的稳定性。

6.方差阈值 (Variance Threshold):

| 可以通过设定方差的阈值，剔除方差较小的特征，因为它们可能携带的信息相对较少。是一种简单而有效的特征选择方法。

7.LDA (linear discriminant analysis):

| 通过最大化类别间差异和最小化类别内差异。

8.使用树模型:

| 随机森林和梯度提升树等决策树模型在处理高维数据时通常表现较好，因为它们可以自动选择重要的特征。

9.特征工程 (Feature Engineering):

| 基于领域知识，设计新的特征，也可以通过组合、转换原始特征来降低维度。

总体而言，L1正则化、主成分分析PCA、递归特征消除RFE、使用树模型和方差阈值等方法相对具有较强的可解释性，因为它们提供了直接解释模型或特征选择过程的信息。然而，对于某些方法（如t-SNE），解释性可能相对较弱，主要用于可视化和聚类。


.. image:: ../../_static/machine_learning/feature_variance.png
	:align: center
	:width: 350


特征选择 
-----------------
怎样选择特征，如何筛选特征

特征选择 https://zhuanlan.zhihu.com/p/32749489 这篇文章有点东西的，解释的很详细，而且可以基于sklearn给出示例。

当数据预处理完成后，我们需要选择有意义的特征输入机器学习的算法和模型进行训练。通常来说，从两个方面考虑来选择特征：

| •	特征是否发散：如果一个特征不发散，例如方差接近于0，也就是说样本在这个特征上基本上没有差异，这个特征对于样本的区分并没有什么用。
| •	特征与目标的相关性：这点比较显见，与目标相关性高的特征，应当优选选择。除移除低方差法外，本文介绍的其他方法均从相关性考虑。

根据特征选择的形式又可以将特征选择方法分为3种：

| •	Filter：过滤法，按照发散性或者相关性对各个特征进行评分，设定阈值或者待选择阈值的个数，选择特征。
| •	Wrapper：包装法，根据目标函数（通常是预测效果评分），每次选择若干特征，或者排除若干特征。
| •	Embedded：嵌入法，先使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据系数从大到小选择特征。类似于Filter方法，但是是通过训练来确定特征的优劣。

特征选择主要有两个目的：
| •	减少特征数量、降维，使模型泛化能力更强，减少过拟合；
| •	增强对特征和特征值之间的理解。

| Filter
| 1. 移除低方差的特征
| 2. 单变量特征选择 (Univariate feature selection)
　　单变量特征选择的原理是分别单独的计算每个变量的某个统计指标，根据该指标来判断哪些指标重要，剔除那些不重要的指标。

| Wrapper
| 3. 递归特征消除 (Recursive Feature Elimination)
　　递归消除特征法使用一个基模型来进行多轮训练，每轮训练后，移除若干权值系数的特征，再基于新的特征集进行下一轮训练。

| Embedded
| 4. 使用SelectFromModel选择特征 (Feature selection using SelectFromModel)
| 　　单变量特征选择方法独立的衡量每个特征与响应变量之间的关系，另一种主流的特征选择方法是基于机器学习模型的方法。有些机器学习方法本身就具有对特征进行打分的机制，
或者很容易将其运用到特征选择任务中，例如回归模型，SVM，决策树，随机森林等等。其实Pearson相关系数等价于线性回归里的标准化回归系数。

最大似然/最小二乘
----------------------------------
**最小二乘**，最合理的参数估计量应该使得模型能最好地拟合样本数据，也就是估计值和观测值之差的平方和最小

找一个（组）估计值，使得实际值与估计值之差的平方加总之后的值最小，称为最小二乘。

**最大似然**，就是利用已知的样本结果，反推最有可能（最大概率）导致这样结果的参数值。

用ln把乘法变成加法，且不会改变极值的位置

| 
| 误差服从高斯分布的情况下， 最小二乘法等价于极大似然估计。



为什么先划分训练集和测试集后归一化
----------------------------------------------------
先对数据划分训练集和测试集后归一化和对数据归一化后划分测试集和训练集，两者的区别：

理论上还是应该 **先划分数据集**，然后对训练集做预处理，并且保存预处理的参数， **再用同样的参数处理测试集**

因为划分训练集和测试集就是假设只知道训练集的信息，而认为 **测试集数据是来自未来的，不可得知。如果之前统一做预处理之后再划分的话就利用了测试集的信息**



进程与线程
-------------------
先来个直观的解释。核心是 一个进程可以是多线程 （可以有多条线）

.. image:: ../../_static/machine_learning/进程线程.png
	:align: center


https://www.zhihu.com/question/25532384/answer/1130818664 这个解答说的很好，解释的具体，而且面试题也涉及了

| 核心：
| 进程是资源分配的基本单位；线程是程序执行的基本单位。
| 一个进程可以包含若干个线程。

| 进程/线程如何通信
| 答：进程可以通过管道、套接字、信号交互、共享内存、消息队列等等进行通信；而线程本身就会共享内存，指针指向同一个内容，交互很容易。



概率论
======================


概率论中的常见分布类型
------------------------------------
**三种离散型分布：Bernoulli Distribution伯努利分布、Binomial Distribution二项分布、poisson distribution**

**1. Bernoulli Distribution伯努利分布**

伯努利分布(两点分布/0-1分布)：伯努利试验指的是只有两种可能结果的单次随机试验。若随机变量X的取值为0和1两种情况，且满足概率分布P(X=1)=p, P(X=0)=1-p，则X服从参数为p的伯努利分布。

举例：假设有产品100件，其中正品90件，次品10件。现在随机从这100件中挑选1件，那么他挑选出正品的概率为0.9，即P(X=正品)=p = 0.9

**2.Binomial Distribution二项分布**

二项分布是将一个『只有两种可能结果的实验』重复n次，得到n+1种『最终实验结果』。伯努利分布分布是二项分布的特例，二项分布是0-1分布的n次重复

.. image:: ../../_static/machine_learning/Binomial_distribution_pmf.png
	:width: 400


**3.poisson distribution**

一个单位内(时间、面积、空间)某稀有事件发生K次的概率。  P(X=0),P(X=1),P(X=3),….所有可能的概率共同组成了一个分布，即泊松分布。

.. image:: ../../_static/machine_learning/Poisson_pmf.png
	:width: 400

应用举例：

| 某时间段内，来到某商场的顾客数
| 单位时间内，某网站的点击量
| 一平方米内玻璃上的气泡数


**三种连续型分布Normal distribution正态分布、Uniform distribution均匀分布、Exponential distribution指数分布**

**4.Normal distribution正态分布**

.. image:: ../../_static/machine_learning/normal_distribution.png
	:width: 300

.. image:: ../../_static/machine_learning/1280px-Normal_Distribution_PDF.png
	:width: 500


**5.Uniform distribution均匀分布**

.. image:: ../../_static/machine_learning/uniform_distribution.png
	:width: 400

.. image:: ../../_static/machine_learning/uniform_distribution2.png
	:width: 400



**6.Exponential distribution指数分布**

.. image:: ../../_static/machine_learning/exponential_distribution.png
	:width: 300

.. image:: ../../_static/machine_learning/exponential_distribution2.png
	:width: 300


中心极限定理
----------------------------
中心极限定理的准定义是：

中心极限定理（CLT）指出，如果样本量足够大，则变量**均值**的采样分布将近似于正态分布，而与该变量在总体中的分布无关。

中心极限定理意味着即使数据分布不是正态的，从中抽取的样本均值的分布也是正态的。

大数定律
------------------
在试验不变的条件下，重复试验多次，随机事件的频率近似于它的概率


.. image:: ../../_static/machine_learning/largenumber.png
	:align: center

.. image:: ../../_static/machine_learning/largenumber2.png
	:align: center
	
	
相比较弱大数定律，强大数定律表征着当数列样本量增大后，它再也不会超出虚线所表示的边界，也就是超出这个边界的概率就是0了。这个就叫做强大数定律的处处收敛。


投骰子连续两次是6就停止，求投掷的次数的期望
-----------------------------------------------------
.. image:: ../../_static/machine_learning/骰子.png
	:align: center

投硬币连续两次是正面就停止，求投掷的次数的期望
-----------------------------------------------------
.. image:: ../../_static/machine_learning/硬币1.png
	:align: center
	:width: 400
	
注意： 扔到两次，都是正面，结束，则是0.25*2  这里乘了2！！

抛硬币直到出现连续N次正面为止的期望
---------------------------------------------------------
.. image:: ../../_static/machine_learning/硬币2.png
	:align: center



