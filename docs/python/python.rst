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




生成器和迭代器
----------------------
https://www.jianshu.com/p/dcc4c1af63c7

http://www.techweb.com.cn/cloud/2020-07-27/2798448.shtml

生成器：iter() 和 next()

迭代器： yield

省内存

*arg与**kwargs参数的用法
----------------------------------------------
https://www.cnblogs.com/xujiu/p/8352635.html

*arg表示任意多个无名参数，类型为tuple;**kwargs表示关键字参数，为dict


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
    
**几个坑的地方要特别注意：**

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
| 如果这次面试的感觉好，就是：请问入职以后对我们有什么系统的培训吗？
| 如果感觉不好，能否对我今天的面试或者之后的学习提出一些建议？
|  **记得问部门剩余多少HC，我多久能收到通知**

5.多面，多练手，才不会那么紧张

6.自我介绍和项目介绍一定要准备好。之前的一分钟自我介绍太短了，导致后面很被动。

7.要很有自信，就像是在和老板讲故事一样，自己说出来的话都没底气，别人怎么会相信你。
不要战战兢兢的像是小学的时候老师抽查你背课文一样，就当跟同学之间的聊天和探讨吹牛皮。

8.面试要经常总结和做面经，不然会在一个坑里一次又一次的跌倒。

9.多去和师兄同学讨论，请教。不要闭门造车

笔试做题和面试做题
------------------------
1. 输入输出要搞明白，line那个变量没有定义这种事情不要再发生了（我经常搞出这种变量未定义，超边界的事情）。例题1和例题2多看看

2. 笔试的优点在于：可以用愚蠢的暴力法去得一个基础分数。可以一个个的去尝试。比如某公司的某个跳台阶的题目，题目没描述清楚，那么我们一个个的去尝试前几个值，能把他的分布找出来

3. 笔试的缺点在于，如果出了任何的bug，是得不了分的。而且解释的机会都没有。而且不能print 的debug

4. print还是return，py2还是py3 一定要看清楚。而且，某公司让你取100000007%的模，那就一定要取！

5. 脑子不能僵硬，该那啥那啥。选择题和编程题都是。

说出自己的三个缺点
-------------------------------------------
自己有充分的理论知识，但是也还需要在工作中得到更进一步的锻炼。

追求完美，比如在学习有足够的时间让自己做到处处完美，但是在实际的工作应该以目标为重，适当的接受一些不那么重要的瑕疵。

心肠软，面对同事的请求帮助或者要求帮忙分担的时候，不会拒绝，所以在工作中能快速跟同事搞好关系，但是适度的拒接其实对自己有好处，应该以自己的工作为重点。

性子急，布置的任务喜欢赶早不赶晚的完成，很讨厌拖拖拉拉。


这种问题真的很无聊很恶心..但是还是要提前准备一下.....反正他恶心你你就恶心他....挑这种不痛不痒的回答