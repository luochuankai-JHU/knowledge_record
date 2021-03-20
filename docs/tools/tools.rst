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

| •	   =	   等于
| •   	<>	   不等于。注释：在 SQL 的一些版本中，该操作符可被写成 !=
| •	   >	   大于
| •	   <   	小于
| •	   >=	大于等于
| •	   <=   	小于等于
| •	   BETWEEN   	在某个范围内
| •	   LIKE   	搜索某种模式
| •	   IN   	指定针对某个列的多个可能值


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

| •	INNER JOIN：如果表中有至少一个匹配，则返回行
| •	LEFT JOIN：即使右表中没有匹配，也从左表返回所有的行
| •	RIGHT JOIN：即使左表中没有匹配，也从右表返回所有的行
| •	FULL JOIN：只要其中一个表中存在匹配，则返回行


