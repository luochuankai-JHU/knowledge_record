.. knowledge_record documentation master file, created by
   sphinx-quickstart on Tue July 4 21:15:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

******************
一些零散记录
******************

一些学习资料
=========================

李航 统计学习方法

台湾大学  林轩田  机器学习基石  进阶

深度学习   台湾大学 李宏毅

深度学习  CV NLP   斯坦福 CS231  

机器学习、深度学习都有  莫烦  python。 非常入门但是讲的很好

西瓜书

一个做的很好的GitHub网站，里面总结了面试题
https://github.com/DarLiner/Algorithm_Interview_Notes-Chinese




零碎
================

【文档】使用Sphinx + reST编写文档
--------------------------------------
https://www.cnblogs.com/zzqcn/p/5096876.html#_label7

生成a到z，判断是否是数字，判断是否是字母
---------------------------------------

生成a到z::

    num2char = dict()
    for i in range(26):
        num2char[i] = chr(ord("a")+i)

string.isdigit()

string.isalpha()























知识图谱实体链接：一份“由浅入深”的综述
------------------------------------------
https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/103900689


.. image:: ../../_static/python/entity_linking.png
    :align: center


被遮住的部分：

1. Candidate Entity Generation (CEG) 

最重要的方法：Name Dictionary ( {mention: entity} ) 

哪些别名：首字母缩写、模糊匹配、昵称、拼写错误等。 

构建方法： 

Wikipedia (Redirect pages, Disambiguation pages, Hyperlinks)； 

基于搜索引擎：调 google api，搜 mention。若前 m 个有 wiki entity，建立 map； 

Heuristic Methods； 

人工标注、用户日志。 




3. 还有个小问题：Unlinkable Mention Prediction 

除了上面的两大模块，还有一个小问题，就是如何拒识掉未知实体，毕竟你不可能建立一个能穷举万物的 KB。这就涉及到 Unlinkable Mention Prediction，不是很复杂，一般就三种做法： 

NIL Threshold: 通过一个置信度的阈值来卡一下； 

Binary Classification: 训练一个二分类的模型，判断 Top-rankeded Entity 是否真的是文中的 mention 想要表达的实体； 

Rank with NIL: 在 rank 的时候，在候选实体中加入 NIL Entity。 

一般就阈值卡一下就好了，不是太大的问题。但如果具体的场景是做 KB Population 且实体还不是很全的时候，就需要重点关注一下了。


2. Type Classifier: 给定 mention 和 text，输出 mention 对应实体的 type； 


