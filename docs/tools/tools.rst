.. knowledge_record documentation master file, created by
   sphinx-quickstart on Tue July 4 21:15:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

******************
实用工具
******************

SQL语法
=====================

大致概念与资源
--------------------------------------------------------
大概的概念   https://www.bilibili.com/video/BV1q441167Su?from=search&seid=17968570397124288241  

.. image:: ../../_static/tools/need.png
    :align: center
    :width: 550
    
    
.. image:: ../../_static/tools/SQL命令总.png
    :align: center
    :width: 550
    
基本的操作  https://www.bilibili.com/video/BV1ZJ411i7YM?from=search&seid=17968570397124288241

SQL 常用命令及练习--之一     https://zhuanlan.zhihu.com/p/37110401


开始
--------------------

开始::

    mysql> use RUNOOB;//选择数据库
    Database changed

    mysql> set names utf8;//设置使用的字符集
    Query OK, 0 rows affected (0.00 sec)

    mysql> SELECT * FROM Websites;//读取数据表的信息
    +----+--------------+---------------------------+-------+---------+
    | id | name         | url                       | alexa | country |
    +----+--------------+---------------------------+-------+---------+
    | 1  | Google       | https://www.google.cm/    | 1     | USA     |
    | 2  | 淘宝          | https://www.taobao.com/   | 13    | CN      |
    | 3  | 菜鸟教程      | http://www.runoob.com/    | 4689  | CN      |
    | 4  | 微博          | http://weibo.com/         | 20    | CN      |
    | 5  | Facebook     | https://www.facebook.com/ | 3     | USA     |
    +----+--------------+---------------------------+-------+---------+
    5 rows in set (0.01 sec)
    

SQL SELECT 语句
----------------------------

| SELECT 语句用于从数据库中选取数据。结果被存储在一个结果表中，称为结果集。


SQL SELECT 语法::

    SELECT column_name,column_name
    FROM table_name;

    SELECT * FROM table_name;


eg::

    SELECT name,country FROM Websites;//选择name、country列
    SELECT * FROM Websites;


SQL SELECT DISTINCT 语句
-------------------------------------

| 在表中，一个列可能会包含多个重复值，有时您也许希望仅仅列出不同（distinct）的值。
| DISTINCT 关键词用于返回唯一不同的值。

SQL SELECT DISTINCT 语法::

    SELECT DISTINCT column_name,column_name
    FROM table_name;

eg::

    SELECT DISTINCT country FROM Websites;
    //从 "Websites" 表的 "country" 列中选取唯一不同的值，也就是去掉 "country" 列重复值
    
SQL WHERE 子句
-------------------------------------

WHERE 子句用于提取那些满足指定标准的记录。

SQL WHERE 语法::

    SELECT column_name,column_name
    FROM table_name
    WHERE column_name operator value;

eg::

    SELECT * FROM Websites WHERE country='CN';//文本字段用引号
    SELECT * FROM Websites WHERE id=1;//数值字段不用引号

    SELECT name, population FROM world
      WHERE name IN ('Luxembourg', 'Mauritius', 'Samoa');
    SELECT name, area FROM world
      WHERE area BETWEEN 250000 AND 300000
      
      
WHERE 子句中的运算符
-------------------------------

| •       =       等于
| •       <>       不等于。注释：在 SQL 的一些版本中，该操作符可被写成 !=
| •       >       大于
| •       <       小于
| •       >=    大于等于
| •       <=       小于等于
| •       BETWEEN       在某个范围内
| •       LIKE       搜索某种模式
| •       IN       指定针对某个列的多个可能值


SQL AND & OR 运算符
-------------------------------
| 如果第一个条件和第二个条件都成立，则 AND 运算符显示一条记录。
| 如果第一个条件和第二个条件中只要有一个成立，则 OR 运算符显示一条记录。

eg::

    SELECT * FROM Websites
    WHERE country='CN'
    AND alexa > 50;//选择CN为country的alexa大于50的列

    SELECT * FROM Websites
    WHERE country='USA'
    OR country='CN';

    SELECT * FROM Websites
    WHERE alexa > 15
    AND (country='CN' OR country='USA');


SQL ORDER BY 关键字
--------------------------
| ORDER BY 关键字用于对结果集按照一个列或者多个列进行排序。
| ORDER BY 关键字默认按照升序对记录进行排序。如果需要按照降序对记录进行排序，您可以使用 DESC 关键字。

SQL ORDER BY 语法::

    SELECT column_name,column_name
    FROM table_name
    ORDER BY column_name,column_name ASC|DESC;

    SELECT * FROM Websites
    ORDER BY alexa;//按照alexa列升序排列

    SELECT * FROM Websites
    ORDER BY alexa DESC;//按照alexa列降序排列

    SELECTSE  * FROM Websites
    ORDER BY country,alexa;//先按照第一个column name排序，再按照第二个column name排序



SQL INSERT INTO 语句
--------------------------
INSERT INTO 语句用于向表中插入新记录。

SQL INSERT INTO 语法

INSERT INTO 语句可以有两种编写形式。

第一种形式无需指定要插入数据的列名，只需提供被插入的值即可::

    INSERT INTO table_name
    VALUES (value1,value2,value3,...);

    INSERT INTO Websites (name, url, alexa, country)
    VALUES ('百度','https://www.baidu.com/','4','CN');
    
第二种形式需要指定列名及被插入的值::

    INSERT INTO table_name (column1,column2,column3,...)
    VALUES (value1,value2,value3,...);

    INSERT INTO Websites (name, url, country)
    VALUES ('stackoverflow', 'http://stackoverflow.com/', 'IND');//alexa未指定


SQL UPDATE 语句
--------------------------

UPDATE 语句用于更新表中已存在的记录。

SQL UPDATE 语法::

    UPDATE table_name
    SET column1=value1,column2=value2,...
    WHERE some_column=some_value;

eg::

    UPDATE Websites 
    SET alexa='5000', country='USA' 
    WHERE name='菜鸟教程';


SQL DELETE 语句
--------------------------
DELETE 语句用于删除表中的行。

SQL DELETE 语法::

    DELETE FROM table_name
    WHERE some_column=some_value;

eg::

    DELETE FROM Websites
    WHERE name='百度' AND country='CN';



IN 操作符
------------------------
IN 操作符允许您在 WHERE 子句中规定多个值。

SQL IN 语法::

    SELECT column_name(s)
    FROM table_name
    WHERE column_name IN (value1,value2,...);

    SELECT * FROM Websites
    WHERE name IN ('Google','菜鸟教程');


SQL LIKE 操作符
----------------------------
LIKE 操作符用于在 WHERE 子句中搜索列中的指定模式。

SQL LIKE 语法::

    SELECT column_name(s)
    FROM table_name
    WHERE column_name LIKE pattern;

    SELECT * FROM Websites
    WHERE name LIKE 'G%';//以G开头
    
    SELECT * FROM Websites
    WHERE name LIKE '%k';//以k结尾
    
    SELECT * FROM Websites
    WHERE name LIKE '%oo%';//包含oo
    
    SELECT * FROM Websites
    WHERE name NOT LIKE '%oo%';//不包含oo
    
    SELECT * FROM Websites
    WHERE name LIKE '_oogle';
    
    SELECT * FROM Websites
    WHERE name REGEXP '^[GFs]';//选取 name 以 "G"、"F" 或 "s" 开始的所有网站
    
    SELECT * FROM Websites
    WHERE name REGEXP '^[A-H]';//选取 name 不以 A 到 H 字母开头的网站



SQL BETWEEN 操作符
--------------------------------
BETWEEN 操作符选取介于两个值之间的数据范围内的值。这些值可以是数值、文本或者日期。

SQL BETWEEN 语法::

    SELECT column_name(s)
    FROM table_name
    WHERE column_name BETWEEN value1 AND value2;

    SELECT * FROM Websites
    WHERE alexa BETWEEN 1 AND 20;//选取 alexa 介于 1 和 20 之间的所有网站
    
    SELECT * FROM Websites
    WHERE alexa NOT BETWEEN 1 AND 20;
    
    SELECT * FROM Websites
    WHERE (alexa BETWEEN 1 AND 20)
    AND NOT country IN ('USA', 'IND');//选取alexa介于 1 和 20 之间但 country 不为 USA 和 IND 的所有网站
    
    SELECT * FROM Websites
    WHERE name BETWEEN 'A' AND 'H';//选取 name 以介于 'A' 和 'H' 之间字母开始的所有网站
    
    SELECT * FROM Websites
    WHERE name NOT BETWEEN 'A' AND 'H';//选取 name 不介于 'A' 和 'H' 之间字母开始的所有网站
    
    SELECT * FROM access_log
    WHERE date BETWEEN '2016-05-10' AND '2016-05-14';//选取 date 介于 '2016-05-10' 和 '2016-05-14' 之间的所有访问记录


SQL 别名
---------------------
通过使用 SQL，可以为表名称或列名称指定别名。基本上，创建别名是为了让列名称的可读性更强。

列的 SQL 别名语法::

    SELECT column_name AS alias_name
    FROM table_name;

表的 SQL 别名语法::

    SELECT column_name(s)
    FROM table_name AS alias_name;

    SELECT name, CONCAT(url, ', ', alexa, ', ', country) AS site_info
    FROM Websites;//我们把三个列（url、alexa 和 country）结合在一起，
    并创建一个名为 "site_info" 的别名

    SELECT w.name, w.url, a.count, a.date 
    FROM Websites AS w, access_log AS a 
    WHERE a.site_id=w.id and w.name="菜鸟教程";//我们使用 "Websites" 和 "access_log" 表，
    并分别为它们指定表别名 "w" 和 "a"
    
    
SQL JOIN
---------------------------------
SQL JOIN 子句用于把来自两个或多个表的行结合起来，基于这些表之间的共同字段。

最常见的 JOIN 类型：SQL INNER JOIN（简单的 JOIN）。 

SQL INNER JOIN 从多个表中返回满足 JOIN 条件的所有行::

    SELECT Websites.id, Websites.name, access_log.count, access_log.date
    FROM Websites
    INNER JOIN access_log
    ON Websites.id=access_log.site_id;//"Websites" 表中的 "id" 列指向 "access_log" 表中的字段 "site_id"。
    上面这两个表是通过 "site_id" 列联系起来的

| •    INNER JOIN：如果表中有至少一个匹配，则返回行
| •    LEFT JOIN：即使右表中没有匹配，也从左表返回所有的行
| •    RIGHT JOIN：即使左表中没有匹配，也从右表返回所有的行
| •    FULL JOIN：只要其中一个表中存在匹配，则返回行


SQL INNER JOIN 关键字
----------------------------------------
INNER JOIN 关键字在表中存在至少一个匹配时返回行。

SQL INNER JOIN 语法::

    SELECT column_name(s)
    FROM table1
    INNER JOIN table2
    ON table1.column_name=table2.column_name;

或::

    SELECT column_name(s)
    FROM table1
    JOIN table2
    ON table1.column_name=table2.column_name;


.. image:: ../../_static/tools/innerjoin.png
    :align: center
    :width: 150
    
    
    
SQL UNION 操作符
--------------------------
UNION 操作符用于合并两个或多个 SELECT 语句的结果集。请注意，UNION 内部的每个 SELECT 语句必须拥有相同数量的列。列也必须拥有相似的数据类型。同时，每个 SELECT 语句中的列的顺序必须相同。

SQL UNION 语法::

    SELECT column_name(s) FROM table1
    UNION
    SELECT column_name(s) FROM table2;

注释：默认地，UNION 操作符选取不同的值。如果允许重复的值，请使用 UNION ALL。

eg::

    SELECT country FROM Websites
    UNION
    SELECT country FROM apps
    ORDER BY country;

//UNION 不能用于列出两个表中所有的country。如果一些网站和APP来自同一个国家，每个国家只会列出一次。UNION 只会选取不同的值。请使用 UNION ALL 来选取重复的值！

SQL UNION ALL 语法
----------------------------------------

eg::

    SELECT column_name(s) FROM table1
    UNION ALL
    SELECT column_name(s) FROM table2;

eg::

    SELECT country FROM Websites
    UNION ALL
    SELECT country FROM apps
    ORDER BY country;//使用 UNION ALL 从 "Websites" 和 "apps" 表中选取所有的country（也有重复的值）

    SELECT country, name FROM Websites
    WHERE country='CN'
    UNION ALL
    SELECT country, app_name FROM apps
    WHERE country='CN'
    ORDER BY country;

//下面的 SQL 语句使用 UNION ALL 从 "Websites" 和 "apps" 表中选取所有的中国(CN)的数据SQL NULL 值如果表中的某个列是可选的，那么我们可以在不向该列添加值的情况下插入新记录或更新已有的记录。这意味着该字段将以 NULL 值保存。

SQL IS NULL
--------------------------
我们如何仅仅选取在 "Address" 列中带有 NULL 值的记录呢？

我们必须使用 IS NULL 操作符::

    SELECT LastName,FirstName,Address FROM Persons
    WHERE Address IS NULL

SQL IS NOT NULL
-------------------------------------
我们如何仅仅选取在 "Address" 列中不带有 NULL 值的记录呢？

我们必须使用 IS NOT NULL 操作符::

    SELECT LastName,FirstName,Address FROM Persons
    WHERE Address IS NOT NULL
    
    
GROUP BY 语句
-----------------------
GROUP BY 语句用于结合聚合函数，根据一个或多个列对结果集进行分组。

SQL GROUP BY 语法::

    SELECT column_name, aggregate_function(column_name)
    FROM table_name
    WHERE column_name operator value
    GROUP BY column_name;

GROUP BY 简单应用，统计 access_log 各个 site_id 的访问量::

    mysql> SELECT * FROM access_log;
    +-----+---------+-------+------------+
    | aid | site_id | count | date       |
    +-----+---------+-------+------------+
    |   1 |       1 |    45 | 2016-05-10 |
    |   2 |       3 |   100 | 2016-05-13 |
    |   3 |       1 |   230 | 2016-05-14 |
    |   4 |       2 |    10 | 2016-05-14 |
    |   5 |       5 |   205 | 2016-05-14 |
    |   6 |       4 |    13 | 2016-05-15 |
    |   7 |       3 |   220 | 2016-05-15 |
    |   8 |       5 |   545 | 2016-05-16 |
    |   9 |       3 |   201 | 2016-05-17 |
    +-----+---------+-------+------------+
    9 rows in set (0.00 sec)



    SELECT site_id, SUM(access_log.count) AS nums
    FROM access_log GROUP BY site_id;
    
执行以上 SQL 输出结果如下：

.. image:: ../../_static/tools/groupby1.png
    :width: 500









hadoop常用命令
==========================

大部分hadoop命令跟Linux命令相同，只是在使用时需要加上hadoop fs前缀。

各命令请看官方文档： Hadoop Shell命令  http://hadoop.apache.org/docs/r1.0.4/cn/hdfs_shell.html


hadoop HDFS MapReduce afs简介
------------------------------------------

hadoop：一个由Apache基金会所开发的分布式系统基础架构。核心包括两部分：HDFS和mapreduce。

HDFS：Hadoop Distributed File System，hadoop实现的一个分布式文件系统，用于海量数据的存储。

MapReduce：hadoop使用的计算框架，用于执行对海量数据的高速计算。

AFS：Advanced/Amazing File System，是百度的第二代超大规模文件系统，可以作为其他存储系统的下层，托管所有的离线存储资源，提供存储服务化能力。

ML-arch离线服务的存储与运算使用afs集群与mapreduce计算框架，关于hadoop与mapreduce的详细介绍参见hadoop用户手册。


hadoop fs、hadoop dfs、hdfs dfs的区别
----------------------------------------------------
fs与dfs对于hadoop来说是两个不同的shell，两者的区别在于fs可以操作所有的文件系统，而dfs只能操作HDFS文件系统。


ls命令
----------------------------------------------------
使用方法：hadoop fs -ls ***

对于文件***，返回文件信息（权限 副本数 用户ID 组ID 文件大小 修改日期 修改时间 文件名）

对于目录***，列出目录文件（权限 副本数 用户ID 组ID 0（文件）/子目录文件数 修改日期 修改时间 文件名/子目录名）

e.g::

    hadoop fs -ls afs://xingtian.afs.baidu.com:portname/path 列出path目录下所有文件的上述信息



cat命令
----------------------------------------------------
hadoop fs -cat ***：查看***文件内容（可以搭配grep/wc/count等命令一起使用）

e.g::

    hadoop fs -cat afs://xingtian.afs.baidu.com:portname/path/filename | grep 'index' | head -n 100

    查看文件filename（只显示带有index字符串的前100行）



mkdir命令
----------------------------------------------------
hadoop fs -mkdir <paths>：创建目录（一般创建目录需要有对应目录的权限）

e.g::

    hadoop fs -mkdir  afs://xingtian.afs.baidu.com:portname/path/test 在path路径下创建新文件夹test

rmr命令
------------------------
hadoop fs -rmr ***：删除文件或目录（可能会需要权限，慎用此命令）

e.g::

    hadoop fs -rmr afs://xingtian.afs.baidu.com:portname/path/test 删除文件test或者文件夹test

get命令
------------------------
hadoop fs -get <afs_paths> <localdst>：复制文件到本地文件系统

e.g::

    hadoop fs -get afs://xingtian.afs.baidu.com:portname/path/test -/example

    从afs复制文件（或目录）test到本地-/example文件夹

put命令
------------------------
hadoop fs -put *** <afs-paths> ：复制本地文件***到afs系统

e.g::

    hadoop fs -put /home/work/20180703/ afs://xingtian.afs.baidu.com:portname/path/test/data

复制本地当前文件夹20180703到集群data目录，如果目标data目录不存在，则会创建data目录并把/home/work/test/下面的文件拷贝到data目录下（不保留20180703文件夹）。

即如果20180703目录下有文件test.txt，而目标路径无data目录，则结果会是/test/data/test.txt，目标路径有data目录，put的结果才会是/test/data/20180703/test.txt，

这里要注意，否则会跟预期结果不一样。

权限问题（ugi）
------------------------
当对非当前用户组的文件进行操作时，会遇到权限问题，解决办法为在 fs 和命令中添加

-D hadoop.job.ugi=username,groupname以新的用户ID和组ID去访问目标路径文件。

e.g::

    hadoop fs -D hadoop.job.ugi=username,groupname ls ***

杀死任务（kill命令）
------------------------
hadoop job <ugi> <tracker> -kill <job id>：kill tracker集群中正运行的job

e.g::

    hadoop job -Dhadoop.job.ugi=***,***  -Dmapred.job.tracker=szwg-wuge-job.szwg.dmop.baidu.com:54311 -kill job_20190501005919_3804195

杀死集群szwg-wuge-job.szwg.dmop.baidu.com:54311中job job_20190501005919_3804195。

更改任务优先级
------------------------
hadoop job <ugi> <tracker> -set-priority <job id> <priority>

e.g::

    hadoop job -Dhadoop.job.ugi=***,*** -Dmapred.job.tracker=szwg-wuge-job.szwg.dmop.baidu.com:54311 -set-priority job_20190501005919_3789481 VERY_HIGH

计算文件夹/文件大小（du/dus命令）
------------------------------------------------
hadoop fs -du <afs-paths-dir>  ：列出文件夹中所有文件的大小

hadoop fs -dus <afs-paths-dir>：列出文件夹的大小

touchz命令
------------------------
hadoop fs -touchz <afs_paths>：创建一个0字节的空文件，成功返回0，失败返回 -1.

e.g::

    hadoop fs -touchz afs://xingtian.afs.baidu.com:9902/user/feed/mlarch/lijunjun/test_file

    在afs://xingtian.afs.baidu.com:9902/user/feed/mlarch/lijunjun目录下创建空文件test_file。

集群间copy数据(distcp)
---------------------------

命令

/home/work/pingo/tool/hmpclient/bin/hadoop distcp -Dfs.default.name=<任务default集群> -Dhadoop.job.ugi=<任务ugi> -D mapred.job.queue.name=<任务队列> -D mapred.job.tracker=<任务集群tracker> -D dfs.replication=3 -D mapred.job.map.capacity=5000 -D mapred.job.priority=HIGH -su src_ugi -du dest_ugi -update src_path dest_path
用例

hadoop distcp  -Dfs.default.name=afs://xingtian.afs.baidu.com:9902 -Dhadoop.job.ugi=mlarch,****** -D mapred.job.queue.name=feed-mlarch -D mapred.job.tracker=yq01-xingtian-job.dmop.baidu.com:54311  -D dfs.replication=3 -D mapred.job.map.capacity=5000 -D mapred.job.priority=HIGH -su mlarch,****** -du mlarch,****** -update afs://xingtian.afs.baidu.com:9902/user/feed/mlarch/ctr-logmerge/baipai_video_sample/20200521/ afs://shaolin.afs.baidu.com:9902/user/mlarch/ctr-logmerge/baobaozhidao_sample/20200520/13


python在hadoop下编写map-reduce示例
==========================================
使用python在hadoop下编写map-reduce https://blog.csdn.net/laobai1015/article/details/103086737

Hadoop Streaming提供了一个便于进行MapReduce编程的工具包，使用它可以基于一些可执行命令、脚本语言或其他编程语言来实现Mapper和 Reducer，从而充分利用Hadoop并行计算框架的优势和能力，来处理大数据。

部署hadoop环境，这点可以参考 http://www.powerxing.com/install-hadoop-in-centos/

部署hadoop完成后，需要下载hadoop-streaming包，这个可以到 http://www.java2s.com/Code/JarDownload/hadoop-streaming/hadoop-streaming-0.23.6.jar.zip 去下载，
或者访问 http://www.java2s.com/Code/JarDownload/hadoop-streaming/
选择最新版本，千万不要选择source否则后果自负，选择编译好的jar包即可，放到/usr/local/hadoop目录下备用

数据：在阿里的天池大数据竞赛网站下载了母婴类购买统计数据，记录了900+个萌萌哒小baby的购买用户名、出生日期和性别信息，天池的地址https://tianchi.shuju.aliyun.com/datalab/index.htm

数据是一个csv文件，结构如下：

用户名,出生日期,性别（0女，1男，2不愿意透露性别）

比如：415971,20121111,0（数据已经脱敏处理）

下面我们来试着统计每年的男女婴人数

接下来开始写mapper程序mapper.py，由于hadoop-streaming是基于Unix Pipe的，数据会从标准输入sys.stdin输入，所以输入就写sys.stdin::

    #!/usr/bin/python
    # -*- coding: utf-8 -*-
     
    import sys
     
    for line in sys.stdin:
        line = line.strip()
        data = line.split(',')
        if len(data)<3:
            continue
        user_id = data[0]
        birthyear = data[1][0:4]
        gender = data[2]
        print >>sys.stdout,"%s\t%s"%(birthyear,gender)


下面是reduce程序，这里大家需要注意一下，map到reduce的期间，hadoop会自动给map出的key排序，所以到reduce中是一个已经排序的键值对，这简化了我们的编程工作::

    #!/usr/bin/python
    # -*- coding: utf-8 -*-
    import sys
     
    gender_totle = {'0':0,'1':0,'2':0}
    prev_key = False
    for line in sys.stdin:#map的时候map中的key会被排序
        line = line.strip()    
        data = line.split('\t')
        birthyear = data[0]
        curr_key = birthyear
        gender = data[1]
        
        #寻找边界，输出结果
        if prev_key and curr_key !=prev_key:#不是第一次，并且找到了边界
            print >>sys.stdout,"%s year has female %s and male %s"%(prev_key,gender_totle['0'],gender_totle['1'])
            #先输出上一次统计的结果
            prev_key = curr_key
            gender_totle['0'] = 0
            gender_totle['1'] = 0
            gender_totle['2'] = 0#清零
            gender_totle[gender] +=1#开始计数
        else:
            prev_key = curr_key
            gender_totle[gender] += 1
    #输出最后一行
    if prev_key:
        print >>sys.stdout,"%s year has female %s and male %s"%(prev_key,gender_totle['0'],gender_totle['1'])

接下来就是将样本和mapper reducer上传到hdfs中并执行了

可以先这样测试下python脚本是否正确::

    cat sample.csv | python mapper.py | sort -k1,1 | python reducer.py > result.log

首先要在hdfs中创建相应的目录，为了方便，我将一部分hadoop命令做了别名::

    alias stop-dfs='/usr/local/hadoop/sbin/stop-dfs.sh'
    alias start-dfs='/usr/local/hadoop/sbin/start-dfs.sh'
    alias dfs='/usr/local/hadoop/bin/hdfs dfs'
    echo "alias stop-dfs='/usr/local/hadoop/sbin/stop-dfs.sh'" >> /etc/profile
    echo "alias start-dfs='/usr/local/hadoop/sbin/start-dfs.sh'" >> /etc/profile
    echo "alias dfs='/usr/local/hadoop/bin/hdfs dfs'" >> /etc/profile


启动hadoop后，先创建一个用户目录

hadoop fs -mkdir /user/root/input

然后将样本上传到此目录中

hadoop fs -put ./sample.csv /user/root/input

接下来将mapper.py和reducer.py上传到服务器上，切换到上传以上两个文件的目录

然后就可以执行了::

    hadoop jar /usr/local/hadoop/hadoop-streaming-0.23.6.jar \
    -D mapred.job.name="testhadoop" \
    -D mapred.job.queue.name=testhadoopqueue \
    -D mapred.map.tasks=50 \
    -D mapred.min.split.size=1073741824 \
    -D mapred.reduce.tasks=10 \
    -D stream.num.map.output.key.fields=1 \
    -D num.key.fields.for.partition=1 \
    -input input/sample.csv \    #样本的路径
    -output output-streaming \   #输出结果的路径，自己定义
    -mapper mapper.py \          #上面写的mapper的脚本
    -reducer reducer.py \        #上面写的reducer的脚本
    -file mapper.py \
    -file reducer.py \
    -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner  

命令的解释：

（1）-input：输入文件路径

（2）-output：输出文件路径

（3）-mapper：用户自己写的mapper程序，可以是可执行文件或者脚本

（4）-reducer：用户自己写的reducer程序，可以是可执行文件或者脚本

（5）-file：打包文件到提交的作业中，可以是mapper或者reducer要用的输入文件，如配置文件，字典等。这个一般是必须有的，
因为mapper和reducer函数都是写在本地的文件中，因此需要将文件上传到集群中才能被执行

（6）-partitioner：用户自定义的partitioner程序

| （7）-D：作业的一些属性（以前用的是-jonconf），具体有：
|               1）mapred.map.tasks：map task数目  
|               设置的数目与实际运行的值并不一定相同，若输入文件含有M个part，而此处设置的map_task数目超过M，那么实际运行map_task仍然是M
|               2）mapred.reduce.tasks：reduce task数目  不设置的话，默认值就为1
|               3）num.key.fields.for.partition=N：shuffle阶段将数据集的前N列作为Key；所以对于wordcount程序，map输出为“word  1”，shuffle是以word作为Key，因此这里N=1

（8）-D stream.num.map.output.key.fields=1 这个是指在reduce之前将数据按前1列做排序，一般情况下可以去掉

 

出现以下字样就是成功了::

    16/08/18 18:35:20 INFO mapreduce.Job:  map 100% reduce 100%
    16/08/18 18:35:20 INFO mapreduce.Job: Job job_local926114196_0001 completed successfully





C++ 入门
==========================================

