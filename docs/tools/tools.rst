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
    :width: 550
    
    
.. image:: ../../_static/tools/SQL命令总.png
    :align: center
    :width: 550
    
基本的操作  https://www.bilibili.com/video/BV1ZJ411i7YM?from=search&seid=17968570397124288241

SQL 常用命令及练习--之一     https://zhuanlan.zhihu.com/p/37110401


SQL语法
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

    SELECT DISTINCT country FROM Websites;//从 "Websites" 表的 "country" 列中选取唯一不同的值，也就是去掉 "country" 列重复值
    
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
=, <>, > ,< ,>= ,<=, BETWEEN, LIKE, IN


