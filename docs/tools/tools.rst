.. knowledge_record documentation master file, created by
   sphinx-quickstart on Tue July 4 21:15:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

******************
实用工具
******************


github操作
===================

命令
--------------

.. image:: ../../_static/tools/git_command.png
    :align: center
    :width: 1300

学习资料
---------------------
| Git入门
| 《猴子都能懂的GIT》
| http://backlogtool.com/git-guide/cn/intro/intro1_1.html （入门篇）
| https://backlog.com/git-tutorial/cn/stepup/stepup1_1.html （高级篇）
| https://backlog.com/git-tutorial/cn/reference/ （常见操作速查）


| 可视化工具gitk，https://lostechies.com/joshuaflanagan/2010/09/03/use-gitk-to-understand-git/
| 图解Git，https://marklodato.github.io/visual-git-guide/index-zh-cn.html



SQL语法
=====================

学习资料
---------------
其实不麻烦，几个小时看一遍这个就懂了 https://www.runoob.com/sql/sql-tutorial.html


经典解读
---------------
以我们最常用的这个模板作为解读

::

    select
        count(*) as pv,
        count(distinct cuid) as uv,
        event_type,
        break_max,
        push_active_level,
        (case when sid like '%30864_6%' then '30864_6' when sid like '%30864_7%' then '30864_7' end) as exp
    from
        default.baiduapp_strategy_ads_push_dimension_info
    where
        event_type in ('ack', 'click')
        and event_day = '20220522'
        and source in  ('1', '2')
        and is_all = '2'
        and tag_type in ('2', '3', '5')
        and (sid like '%30864_6%' or sid like '%30864_7%')
    group by
        event_type,
        exp,
        break_max,
        push_active_level


解读一下。其实就四个部分: select, from, where, group by

首先，拿到一个需求，比如需要统计什么，最重要的是select。因为这会是最后展示结果的结构

from就是从哪个表里找

| where就是一些限制条件。注意，这里的条件可以出现或者不出现在select中，比如这个案例里的source和is_all
| 这里有些in, = , like的判断，看看菜鸟教程就好。sid like '%30864_6%' 就是30864_6前后都可以有其他内容

| group by是因为select的时候有count，也就是计数。在select中除了count的行，都要在这里展示。
| 比如，select中除了count，还有A,B,C。分别有3，4，5种可能。那么交叉一下，返回的结果表里就会有3*4*5种可能。通过count去计数
| 所以select中出现的除了count的行，一定要在group by中出现，不然这个D不知道该怎么办

还有一些join(inner join), left join的语法。还有一些as(别名)的语法看看菜鸟教程就好。



hadoop常用命令
==========================

大部分hadoop命令跟Linux命令相同，只是在使用时需要加上hadoop fs前缀。

各命令请看官方文档： Hadoop Shell命令  http://hadoop.apache.org/docs/r1.0.4/cn/hdfs_shell.html







C++ 入门
==========================================

