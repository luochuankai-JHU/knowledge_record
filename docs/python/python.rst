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

深度学习  CV NLP：斯坦福 CS224   斯坦福 CS231  

机器学习、深度学习都有  莫烦  python。 非常入门但是讲的很好

西瓜书

一个做的很好的GitHub网站，里面总结了面试题
https://github.com/DarLiner/Algorithm_Interview_Notes-Chinese

一些易忘的小代码
========================
生成a到z，判断是否是数字，判断是否是字母
---------------------------------------

生成a到z::

    num2char = dict()
    for i in range(26):
        num2char[i] = chr(ord("a")+i)

string.isdigit()

string.isalpha()

    
lambda
------------------
匿名函数

g = lambda x, y: x + y    
print(g(2, 3))
    
将lambda函数作为参数传递给其他函数。

| • filter函数。此时lambda函数用于指定过滤列表元素的条件。
| 例如filter(lambda x:x%3 == 0,[1,2,3])指定将列表[1,2,3]中能够被3整除的元素过滤出来，其结果是[3]。

| • sorted函数。此时lambda函数用于指定对列表中所有元素进行排序的准则。
| 例如sorted([1, 2, 3, 4, 5, 6, 7, 8, 9], key=lambda x: abs(5-x))将列表[1, 2, 3, 4, 5, 6, 7, 8, 9]按照元素与5距离从小到大进行排序，其结果是[5, 4, 6, 3, 7, 2, 8, 1, 9]。

| • map函数。此时lambda函数用于指定对列表中每一个元素的共同操作。
| 例如map(lambda x: x+1, [1, 2,3])将列表[1, 2, 3]中的元素分别加1，其结果[2, 3, 4]。

| • reduce函数。此时lambda函数用于指定列表中两两相邻元素的结合条件。
| 例如reduce(lambda a, b: '{}, {}'.format(a, b), [1, 2, 3, 4, 5, 6, 7, 8, 9])将列表 [1, 2, 3, 4, 5, 6, 7, 8, 9]中的元素从左往右两两以逗号分隔的字符的形式依次结合起来，其结果是'1, 2, 3, 4, 5, 6, 7, 8, 9'。

sort
-------------
intvs = sorted(intervals, key = lambda x: x[0])  这个老是忘记


re正则
----------------
？？？待总结

enumerate
--------------------------
::

seq = ['one', 'two', 'three']
for i, element in enumerate(seq):
    print i, element

zip
----------------
zip 方法在 Python 2 和 Python 3 中的不同：在 Python 3.x 中为了减少内存，zip() 返回的是一个对象。如需展示列表，需手动 list() 转换。

.. image:: ../../_static/python/zip.png
    :align: center

map
------------------------------------------------
.. image:: ../../_static/python/map.png
    :align: center

作用：括号里前面是函数，后面是作用的数据集

python2里面是直接返回列表，python3里面是返回返回迭代器，list一下就好

我们在笔试题的时候也是这样做的

a = list(map(int,input().strip().split()))

list(map(int, xxx )) 就能把之前的  ['1','3',234] 或者 '11213' 变成 int



*arg与**kwargs参数的用法
----------------------------------------------
https://www.cnblogs.com/xujiu/p/8352635.html

*arg表示任意多个无名参数，类型为tuple;**kwargs表示关键字参数，为dict


any / all
------------------
元素除了是 0、空、FALSE 外都算 TRUE

any() ：如果全为空，0，False，则返回False；如果不全为空，则返回True。

all() ：如果全不为空，则返回True；否则返回False。

.. image:: ../../_static/python/any.png
    :align: center
    :width: 300
    
.. image:: ../../_static/python/all.png
    :align: center
    :width: 300


    
emmmmmm,  () 和 [] 这里有点奇怪.... 但基本上 any 就是逻辑中or，all就是逻辑中 and    

零碎
================

ReadtheDocs、Sphinx
--------------------------------------
【文档】使用Sphinx + reST编写文档  https://www.cnblogs.com/zzqcn/p/5096876.html#_label7

如何用ReadtheDocs、Sphinx快速搭建写书环境  https://www.jianshu.com/p/78e9e1b8553a



tmux的使用
------------------
tmux new -s session-name  新建会话 

tmux ls或ctrl+b s  查看目前有开启的会话 

tmux a -t session-name  接入session-name这个会话 

ctrl+b d或tmux detach  临时断开会话

tmux kill-session -t 1  关闭会话


**窗口操作**
| Ctrl+b PgUp/PgDn/   查看页面之前的输出，按q退出


| Ctrl+b c - (c)reate 生成一个新的窗口
| Ctrl+b n - (n)ext 移动到下一个窗口
| Ctrl+b p - (p)revious 移动到前一个窗口.

| Ctrl+b " - split pane horizontally
| Ctrl+b % - 将当前窗格垂直划分
| Ctrl+b 方向键 - 在各窗格间切换
| Ctrl+b，并且不要松开Ctrl，方向键 - 调整窗格大小
| Ctrl+b 空格键 - 切换窗口内置布局 
| Ctrl+b q - 显示分隔窗口的编号 
| Ctrl+b o - 跳到下一个分隔窗口
| Ctrl+b z - 当前窗口最大化
| Ctrl+b x - 关闭当前窗口
| Ctrl+b & - 确认后退出 tmux 


Linux中查看进程状态信息
--------------------------------

| ps -l   列出与本次登录有关的进程信息；
| ps -aux   查询内存中进程信息；
| ps -aux | grep ***   查询***进程的详细信息；
| top   查看内存中进程的动态信息；
| kill -9 pid   杀死进程。



哈希表的原理
----------------------
利用哈希函数映射,构造出一个键值对。（查找的时候直接根据key去计算储存的位置  洛）




生成器和迭代器
----------------------
https://www.jianshu.com/p/dcc4c1af63c7

http://www.techweb.com.cn/cloud/2020-07-27/2798448.shtml

生成器：iter() 和 next()

迭代器： yield

省内存



Python垃圾回收
-----------------------------
| 一、引用计数
|   Python垃圾回收主要以引用计数为主，分代回收为辅。引用计数法的原理是每个对象维护一个ob_ref，用来记录当前对象被引用的次数，也就是来追踪到底有多少引用指向了这个对象

**当发生以下四种情况的时候，该对象的引用计数器+1**

| 对象被创建 a=14
| 对象被引用 b=a
| 对象被作为参数,传到函数中  func(a)
| 对象作为一个元素，存储在容器中  List={a,”a”,”b”,2}

**与上述情况相对应，当发生以下四种情况时，该对象的引用计数器-1**

| 当该对象的别名被显式销毁时 del a
| 当该对象的引别名被赋予新的对象，a=26
| 一个对象离开它的作用域，例如 func函数执行完毕时，函数里面的局部变量的引用计数器就会减一（但是全局变量不会）
| 将该元素从容器中删除时，或者容器被销毁时。

当指向该对象的内存的引用计数器为0的时候，该内存将会被Python虚拟机销毁

还有一些补充机制


    
    
详解可变、不可变数据类型+引用、深|浅拷贝
----------------------------------------------------------
https://leetcode-cn.com/problems/recover-a-tree-from-preorder-traversal/solution/yu-dao-jiu-shen-jiu-xiang-jie-ke-bian-bu-ke-bian-s/    

| 深拷贝和浅拷贝最根本的区别在于是否真正获取一个对象的复制实体，而不是引用。
| 浅拷贝：只是增加了一个指针指向已存在的内存地址，
| 深拷贝：是增加了一个指针并且申请了一个新的内存，使这个增加的指针指向这个新的内存。



| 可变类型——该对象所指向的内存中的值可以被改变。变量（准确的说是引用）改变后，实际上是其所指的值直接发生改变，并没有发生复制行为，也没有开辟新的出地址，通俗点说就是原地改变。
| 不可变类型——该对象所指向的内存中的值不能被改变。当改变某个变量时候，由于其所指的值不能被改变，相当于把原来的值复制一份后再改变，这会开辟一个新的地址，变量再指向这个新的地址。

可变类型——list, dict, set

不可变类型——int, str, tuple
    
  
python 常用的 string format 形式
-----------------------------------------
| a. %号
|     print("%d" % a)
| b. str.format # python2.6+
|     print("{}".format(a))
| c. f-string  # python3.6+
|     print(f"{a}")
| d. 标准库模板 # python2.4+
|     from string import Template
|     name='EGON'
|     t = Template('Hello $name!')
|     res=t.substitute(name=name)
|     print(res) # Hello EGON!

| 注：总结四种方式的应用场景
| a. 如果格式化的字符串是由用户输入的，那么基于安全性考虑，推荐使用Template
| b. 如果使用的python3.6+版本的解释器，推荐使用f-Stings
| c. 如果要兼容python2.x版本的python解释器，推荐使用str.format
| d. 如果不是测试的代码，不推荐使用%    
    
    
    
linux 操作系统一些命令
---------------------------

| • ls或ll: 列出文件和目录的内容. ll更详细一些 
| • pwd:查看”当前工作目录“的完整路径
| • touch:创建空文件或文件时间戳修改
| • echo:在显示器上显示一段文字，一般起到一个提示的作用
| • mkdir:创建指定的名称的目录，要求创建目录的用户在当前目录中具有写权限，并且指定的目录名不能是当前目录中已有的目录。

| • rmdir:删除空目录
| • nano:文本编辑器
| • vi/vim:文本编辑器，若文件存在则是编辑，若不存在则是创建并编辑
| • shred:用随机值重写覆盖文件，让文件无法恢复
| • cat:连接文件并在标准输出上输出。这个命令常用来显示文件内容，或者将几个文件连接起来显示，或者从标准输入读取内容并显示，它常与重定向符号配合使用。

| • nl:计算文件中行号。nl 可以将输出的文件内容自动的加上行号！其默认的结果与 cat -n 有点不太一样， nl 可以将行号做比较多的显示设计，包括位数与是否自动补齐 。 等等的功能。
| • tac:倒序查看指定文件内容
| • more:类似 cat ，cat命令是整个文件的内容从上到下显示在屏幕上。 more会以一页一页的显示方便使用者逐页阅读，而最基本的指令就是按空白键（space）就往下一页显示，
按 b 键就会往回（back）一页显示，而且还有搜寻字串的功能 。more命令从前向后读取文件，因此在启动时就加载整个文件。
| • less:工具也是对文件或其它输出进行分页显示的工具，应该说是Linux正统查看文件内容的工具，功能极其强大。less 的用法比起 more 更加的有弹性。
在 more 的时候，我们并没有办法向前面翻， 只能往后面看，但若使用了 less 时，就可以使用 [pageup] [pagedown] 等按键的功能来往前往后翻看文件，
更容易用来查看一个文件的内容！除此之外，在 less 里头可以拥有更多的搜索功能，不止可以向下搜，也可以向上搜。
| • grep:文本过滤，模糊查找

| • cut: cut -d : -f 1,4,7 /etc/passwd  --显示etc目录下passwd文件的第1,4,7行
| • sort: 对文件进行排序
| • tr: 字符替换和删除（通常接在管道符后面使用）     echo "HELLO" | tr 'A-Z' 'a-z'  --将大写字母转成小写字母
| • uniq: 删除排序文件中的重复行,通常与sort一起使用
| • tee: 既输出到文件又输出到屏幕    ls | tee 555.log  --将ls看到的内容输出到屏幕并写入到555.log文件里

| • diff: 比较文件之间的差异    vimdiff
| • ln: 创建软硬链接，不带选项为硬链接。
为某一个文件在另外一个位置建立一个同步的链接.当我们需要在不同的目录，用到相同的文件时，我们不需要在每一个需要要的目录下都放一个必须相同的文件，
我们只要在某个固定的目录，放上该文件，然后在 其它的目录下用ln命令链接（link）它就可以，不必重复的占用磁盘空间。
| • type: 查看命令的类型
| • file: 确定文件类型 如果文件系统确定成功，则输出文件类型，输出的文件类型如下：text：文件中只有ASCII码字符，可以将字符终端显示文件内容。executable：文件可以运行。data：其他类型文件，此类文件一般是二进制文件或不能再字符终端上直接显示的文件
| • stat: 查看文件属性   可以显示文件的一些详细信息！！


| • which: which  --查看可执行文件的位置。
| • whereis  --查看文件的位置。
| • locate   --配合数据库查看文件位置。
| • find   --实际搜寻硬盘查询文件名称
| • whatis: whatis cat  --查看命令cat的作用

| • free: 显示内存的使用情况，包括实体内存，虚拟的交换文件内存，共享内存区段，以及系统核心使用的缓冲区等。
| • watch -n 0 nvidia-smi : 看GPU
| • du: 对文件和目录磁盘使用的空间的查看
| • wc: 统计指定文件中的字节数、字数、行数，并将统计结果显示输出。 
| • tail/head -n 1000 -f nohup.out  看文件的后/前 多少行

eg::

    #新建文本
    touch a.txt  #默认权限-rw-rw-r--

    #预览文本
    cat a.txt ，从第一行开始
    tac a.txt #从最后一行开始
    nl a.txt #带行号
    more a.txt #分页，从前往后
    less a.txt #分页，从后往前
    head a.txt #只看头几行
    less a.txt #只看最后几行

    echo "hello" > a.txt #覆盖文件
    echo "hello" >> a.txt #写入文件


pytorch，DDP(DistributedDataParallel)
---------------------------------------------------------------
本来设计主要是为了多机多卡使用，但是单机上也能用

DistributedDataParallel 比DataParallel 快很多，据说能快三倍以上。原因是每个卡都是主卡，...这个具体再看下。

除此之外，还能用 horovod或者 apex 但是都要单独配置

先贴一段自己能跑通的代码。

::

    # import 阶段要多import 这些
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data import DataLoader

    # dataloader 这里要用sampler
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = data.DataLoader(dataset=dataset,
                                 collate_fn=TextCollate(dataset),
                                 pin_memory=True,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=False,
                                 sampler=sampler)
    
    # 初始化这里最恶心
    torch.distributed.init_process_group(backend='nccl')
    # local_rank = args.local_rank
    # torch.cuda.set_device(local_rank)  这样设置好像也可
    local_rank = torch.distributed.get_rank()  # 这样最好
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model.to(device)
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                  device_ids=[local_rank],
                                                  output_device=local_rank,find_unused_parameters=True)
                                                  
    # 如果用到了parser.add_argument，这句话也是需要的
    parser.add_argument('--local_rank', default=-1, type=int)
    
    # 要用shell来跑，按照如下的来写。jupyter的话要另外在代码里面设置别的内容。--nproc_per_node=2因为有两张卡
    python -m torch.distributed.launch --nproc_per_node=2 train_distribute.py
    
几个坑的地方要特别注意：
''''''''''''''''''''''''''''''''''
| 1. 如果pytorch版本只有1.0或者1.1  貌似是没有其他作者写的
| import os
| os.environ['SLURM_NTASKS']          #可用作world size
| os.environ['SLURM_NODEID']          #node id
| os.environ['SLURM_PROCID']          #可用作全局rank
| os.environ['SLURM_LOCALID']         #local_rank
| os.environ['SLURM_STEP_NODELIST']   #从中取得一个ip作为通讯ip
| 这几个功能的？？

| 2. shuffle那里不能用。因为sampler和shuffle是互斥的。所以要自己建立数据集的时候手动shuffle

| 3. find_unused_parameters=True一定要设置，不然坑死！！会报一堆的错，说是有很多数据没有参与反向传播，会变成None，然后都给你打出来了

| 4.初始化这个最恶心。
| 不要初始化端口，不然第一个用了以后第二个会被占用？ 而且world_size，rank 也不要写，不然也会把端口占了？
| world_size: 介绍都是说是进程, 实际就是机器的个数
| rank: 区分主节点和从节点的, 主节点为0, 剩余的为了1-(N-1), N为要使用的机器的数量

| 5.别忘了去掉master_gpu_ids

| 6. 这个可有可无。在使用DataLoader时，别忘了设置pip_memory=true，为什么呢？且看下面的解释，

| 多GPU训练的时候注意机器的内存是否足够(一般为使用显卡显存x2)，如果不够，建议关闭pin_memory(锁页内存)选项。
| 采用DistributedDataParallel多GPUs训练的方式比DataParallel更快一些，如果你的Pytorch编译时有nccl的支持，那么最好使用DistributedDataParallel方式。
| 关于什么是锁页内存：
| pin_memory就是锁页内存，创建DataLoader时，设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。
| 主机中的内存，有两种存在方式，一是锁页，二是不锁页，锁页内存存放的内容在任何情况下都不会与主机的虚拟内存进行交换（注：虚拟内存就是硬盘），
| 而不锁页内存在主机内存不足时，数据会存放在虚拟内存中。显卡中的显存全部是锁页内存,当计算机的内存充足的时候，可以设置pin_memory=True。当系统卡住，
| 或者交换内存使用过多的时候，设置pin_memory=False。因为pin_memory与电脑硬件性能有关，pytorch开发者不能确保每一个炼丹玩家都有高端设备，因此pin_memory默认为False。

https://zhuanlan.zhihu.com/p/97115875 这篇文章讨论到了shuffle 的结果依赖 g.manual_seed(self.epoch) 中的 self.epoch，跑完后再试试

mp的问题，上次拍过棉洲老哥的照片，代码。传到这个GitHub里了，但是没有贴到这上面来。  

多看看官方文档。 好像pytorch1.4还是多少之后就自带apex了

排序问题
-------------------
.. image:: ../../_static/python/sort_all.png
    :align: center

一些排序算法的简单解释

选择排序
''''''''''''''''''''''''''''''''''
每一趟从待排序的数据元素中选出最小（或最大）的一个元素，顺序放在已排好序的数列的最后，直到全部待排序的数据元素排完。

希尔排序
''''''''''''''''''''''''''''''''''
先取一个小于n的证书d1作为第一个增量，把文件的全部记录分成d1组。所有距离为d1的倍数的记录放在同一组中。先在各组内进行直接插入排序，然后取第二个增量d2<d1重复上述的分组和排序，直到所取的增量dt=1，
即所有记录放在同一组中进行直接插入排序为止。该方法实际上是一种分组插入方法。

归并排序
''''''''''''''''''''''''''''''''''
归并排序是把序列递归地分成短序列，递归出口是短序列只有1个元素(认为直接有序)或者2个序列(1次比较和交换)，
然后把各个有序的段序列合并成一个有序的长序列，不断合并直到原序列全部排好序。

堆排序(Heap Sort)
''''''''''''''''''''''''''''''''''
堆排序是一树形选择排序，在排序过程中，将R[1..N]看成是一颗完全二叉树的顺序存储结构，利用完全二叉树中双亲结点和孩子结点之间的内在关系来选择最小的元素。

基数排序
''''''''''''''''''''''''''''''''''
（1）根据数据项个位上的值，把所有的数据项分为10组；

（2）然后对这10组数据重新排列：把所有以0结尾的数据排在最前面，然后是结尾是1的数据项，照此顺序直到以9结尾的数据，这个步骤称为第一趟子排序；

（3）在第二趟子排序中，再次把所有的数据项分为10组，但是这一次是根据数据项十位上的值来分组的。这次分组不能改变先前的排序顺序。也就是说，第二趟排序之后，从每一组数据项的内部来看，数据项的顺序保持不变；

（4）然后再把10组数据项重新合并，排在最前面的是十位上为0的数据项，然后是10位为1的数据项，如此排序直到十位上为9的数据项。

（5）对剩余位重复这个过程，如果某些数据项的位数少于其他数据项，那么认为它们的高位为0。

快速排序
''''''''''''''''''''''''''''''''''
快排的代码在 leetcode那一页有

稳定性
''''''''''''''''''''''''''''''''''
所谓稳定性是指待排序的序列中有两元素相等,排序之后它们的先后顺序不变.假如为A1,A2.它们的索引分别为1,2.则排序之后A1,A2的索引仍然是1和2.

稳定也可以理解为一切皆在掌握中,元素的位置处在你在控制中.而不稳定算法有时就有点碰运气,随机的成分.当两元素相等时它们的位置在排序后可能仍然相同.但也可能不同.是未可知的.

稳定性的用处
''''''''''''''''''''''''''''''''''
我们平时自己在使用排序算法时用的测试数据就是简单的一些数值本身.没有任何关联信息.这在实际应用中一般没太多用处.实际应该中肯定是排序的数值关联到了其他信息,比如数据库中一个表的主键排序,主键是有关联到其他信息.
另外比如对英语字母排序,英语字母的数值关联到了字母这个有意义的信息.

初始状态的影响
''''''''''''''''''''''''''''''''''
| 排序算法不受数据初始状态的影响值得是无论数据是以什么的样的初始状态，那么其最好、平均、最坏的时间复杂度都是一样的，
| （初始数据集排列顺序与比较次数无关）

| 这样的排序算法有堆排序、归并排序、选择排序。
| 他们的时间复杂度为O(nlgn)、O(nlgn)、O(n2)

| 口诀：一堆（堆排序）海归（归并排序）选（选择排序）基友





topK 问题
------------------
坑死了...被很多面试官问过这个问题...这里总结一下。

（1）排序。再取前k个

（2）局部排序。冒泡。冒k个泡，就得到TopK

（3）堆/动态规划。 堆的方法要再看看。  适合处理海量数据  堆 时间复杂度 O(NlogK) 、空间复杂度 O(K)

（4）快速排序改编。 !! **重要**

从数组S中随机取出一个元素，使用一次partition函数，找到该元素对应的位置p，同时将原始数组分成了两个部分S1和S2，显然S1中的元素都小于等于该数，S2中的元素都大于等于该数；此时有三种情况：

| a.如果p等于k，则直接输出S1
| b,如果p大于k,则说明要找的元素全部在S1中,则partition(S1,k)
| c,如果p小于k,则说明要找的元素是S1和S2中的部分元素，则 partition(S2,k-p)
::

    class Solution(object):
        def partition(self,arr,k,low,high):
            i,j = low,high
            p = arr[low]
            while i<j:
                while i<j and arr[j]>=p:
                    j-=1
                while i<j and arr[i]<=p:
                    i+=1
                if i<j:
                    arr[i],arr[j] = arr[j],arr[i]
            arr[low],arr[i] = arr[i],p
            if i==low+k-1:
                return arr[low:low+k]
            elif i>low+k-1:
                return self.partition(arr,k,low,i-1)
            else:
                return arr[low:i+1]+self.partition(arr,k-(i+1-low),i+1,high)
        def getLeastNumbers(self, arr, k):
            """
            :type arr: List[int]
            :type k: int
            :rtype: List[int]
            """
            if k==0:
                return []
            if len(arr)<=k:
                return arr
            return self.partition(arr,k,0,len(arr)-1)

时间空间复杂度？？ 和K有关吗？

找到数组中第k大的元素 (leetcode215. 数组中的第K个最大元素)  跟上面那个有点区别。上面的是topk小，这是第k大
::

        def parti(arr, low, high):
            tmp = arr[low]
            while low<high:
                while low<high and arr[high]>=tmp:
                    high-=1
                arr[low] = arr[high]
                while low<high and arr[low]<=tmp:
                    low +=1
                if low<high:
                    arr[high] = arr[low]
            arr[low] = tmp
            return low

        if not arr or k<=0 or len(arr)<k:
            return []

        low, high, n = 0, len(arr)-1, len(arr)
        index = parti(arr,low, high)
        
        while index != (n-k):
            if index>(n-k):
                high = index-1
                index=parti(arr, low, high)
            else:
                low = index+1
                index=parti(arr, low, high)
        return arr[n-k]


面试总结
==================================
总结一下教训
--------------------

之前什么都不懂....把该犯的错都犯了一遍，这里记录一下深刻的血泪教训....

这哪里像是个正常人做的事啊.......愚蠢到家了


1. 要刷题....真的要刷题，如果一点都没准备，二分查找和树的遍历都写不出，别人凭什么相信你能力强。。。给你机会你不中用啊！

2.不要在什么面试经验都没有的时候从大公司开始投

3.一定要看自己和这个岗位是不是匹配，不用冲着因为是内推所以投个擦边的

| 4.最后面试结束的时候面试官问你，还有没有什么想问的？ 
| 这个职位最紧要任务是什么？如果我有幸入职，您希望我三个月完成哪些工作？
| 这个位的工作业绩如何评估
| 能否对我今天的面试或者之后的学习提出一些建议？
|  **要让面试官介绍一下他们的业务啊！！！！**  面试官谈业务的时候，那支笔拿张纸记一下。然后根据自己的情况去对应着匹配。
|  **记得问部门剩余多少HC，我多久能收到通知**  这个他肯定不会正面回复你，但是你可以看看他的态度
|  **能提前来实习，有稍微差一点的地方或者业务可以提前熟悉**

5.多面，多练手，才不会那么紧张

6.自我介绍和项目介绍一定要准备好。之前的一分钟自我介绍太短了，导致后面很被动。

7.要很有自信，就像是在和老板讲故事一样，自己说出来的话都没底气，别人怎么会相信你。
不要战战兢兢的像是小学的时候老师抽查你背课文一样，就当跟同学之间的聊天和探讨吹牛皮。

8.面试要经常总结和做面经，不然会在一个坑里一次又一次的跌倒。

9.多去和师兄同学讨论，请教。不要闭门造车

10. 当然要去猜面试官到底想问什么，但是不要说出来！！！不要显得自己很聪明的  “啊我猜您想问的呢是XXXX”

笔试做题和面试做题
------------------------
1. 输入输出要搞明白，line那个变量没有定义这种事情不要再发生了（我经常搞出这种变量未定义，超边界的事情）。例题1和例题2多看看

2. 笔试的优点在于：可以用愚蠢的暴力法去得一个基础分数。可以一个个的去尝试。比如某公司的某个跳台阶的题目，题目没描述清楚，那么我们一个个的去尝试前几个值，能把他的分布找出来

3. 笔试的缺点在于，如果出了任何的bug，是得不了分的。而且解释的机会都没有。而且不能print 的debug

4. print还是return，py2还是py3 一定要看清楚。而且，某公司让你取100000007%的模，那就一定要取！

5. 脑子不能僵硬，该那啥那啥。选择题和编程题都是。

6. !!!!输入就用 a = input().  然后记得 a=input().strip().split().   
需要strip，因为有时候输入的东西不干净，前后有空格。然后用split不要用list()....吃过一次亏了,之间把“10”给我分成了["1","0"]

7. 既然可以在自己的本地进行调试。那就一定要在本地调试。用完整的代码，大不了复制粘贴输入输出而已。这样避免用他的调试半天不出结果。而且这样能看见报错。



个税申报
==========================================


投资理财
==========================================
