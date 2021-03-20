.. knowledge_record documentation master file, created by
   sphinx-quickstart on Tue July 4 21:15:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

******************
实用工具
******************

SQL
=====================

大致概念与资源
--------------------------------------------------------
大概的概念   https://www.bilibili.com/video/BV1q441167Su?from=search&seid=17968570397124288241  

.. image:: ../../_static/tools/need.png
    :align: center
    :width: 500
    
    
.. image:: ../../_static/tools/SQL命令总.png
    :align: center
    :width: 500
    
基本的操作  https://www.bilibili.com/video/BV1ZJ411i7YM?from=search&seid=17968570397124288241


SQL语法
--------------------
SQL 常用命令及练习--之一     https://zhuanlan.zhihu.com/p/37110401

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
    







