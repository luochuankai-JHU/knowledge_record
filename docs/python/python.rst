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


python里的map函数，讲一下它的作用和返回值
------------------------------------------------
.. image:: ../../_static/python/map.png
    :align: center

作用：括号里前面是函数，后面是作用的数据集

python2里面是直接返回列表，python3里面是返回返回迭代器，list一下就好


哈希表的原理
----------------------
利用哈希函数映射,构造出一个键值对。（查找的时候直接根据key去计算储存的位置  洛）


海量数据处理面试题
-------------------------
https://www.cnblogs.com/v-July-v/archive/2012/03/22/2413055.html

生成器和迭代器
----------------------
https://www.jianshu.com/p/dcc4c1af63c7

http://www.techweb.com.cn/cloud/2020-07-27/2798448.shtml

生成器：iter() 和 next()

迭代器： yield

省内存

feed流
---------------
https://www.jianshu.com/p/20293026d366

https://www.jianshu.com/p/791817e6f1b0

协同过滤
-----------------
.. image:: ../../_static/python/协同过滤.png
    :align: center
	
	
详解可变、不可变数据类型+引用、深|浅拷贝
----------------------------------------------------------
https://leetcode-cn.com/problems/recover-a-tree-from-preorder-traversal/solution/yu-dao-jiu-shen-jiu-xiang-jie-ke-bian-bu-ke-bian-s/	

可变类型——该对象所指向的内存中的值可以被改变。变量（准确的说是引用）改变后，实际上是其所指的值直接发生改变，并没有发生复制行为，也没有开辟新的出地址，通俗点说就是原地改变。
不可变类型——该对象所指向的内存中的值不能被改变。当改变某个变量时候，由于其所指的值不能被改变，相当于把原来的值复制一份后再改变，这会开辟一个新的地址，变量再指向这个新的地址。

可变类型——list, dict, set

不可变类型——int, str, tuple
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
面试总结
==================================

之前什么都不懂....把该犯的错都犯了一遍，这里记录一下深刻的血泪教训....

这哪里像是个正常人做的事啊.......愚蠢到家了


1. 要刷题....真的要刷题，如果一点都没准备，二分查找和树的遍历都写不出，别人凭什么相信你能力强。。。给你机会你不中用啊！

2.不要在什么面试经验都没有的时候从大公司开始投

3.一定要看自己和这个岗位是不是匹配，不用冲着因为是内推所以投个擦边的

| 4.最后面试结束的时候面试官问你，还有没有什么想问的？ 
| 如果这次面试的感觉好，就是：请问入职以后对我们有什么系统的培训吗？
| 如果感觉不好，能否对我今天的面试或者之后的学习提出一些建议？

5.多面，多练手，才不会那么紧张

6.自我介绍和项目介绍一定要准备好。之前的一分钟自我介绍太短了，导致后面很被动。

7.要很有自信，就像是在和老板讲故事一样，自己说出来的话都没底气，别人怎么会相信你。
不要战战兢兢的像是小学的时候老师抽查你背课文一样，就当跟同学之间的聊天和探讨吹牛皮。

8.面试要经常总结和做面经，不然会在一个坑里一次又一次的跌倒。

9.多去和师兄同学讨论，请教。不要闭门造车



