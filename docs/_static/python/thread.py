# -*- coding:utf-8 -*-
#!/usr/bin/env python
import sys
import zlib 
import json
import copy
import time
import numpy as np
import argparse 
import collections
from threading import Thread, Lock
import jieba
import random
import os
import argparse
sys.dont_write_bytecode = True
import tool_nid_to_nid_info
reload(tool_nid_to_nid_info)
global nid_title_description_dict
global is_video
lock_dict = Lock() 
lock_content = Lock()  
lock_print = Lock()


def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--is_video', type=int, required=True, help='视频or图文的判断')
    args = parser.parse_args()
    return args

# 读字典：nid_title_tag_dict
nid_title_description_dict = dict()
nid_title_tag_dict_path = "../" + folder + "/nid_title_tag_dict"
with open(nid_title_tag_dict_path, "r") as file:
    for line in file:
        arr = line.strip().split()
        if len(arr) < 2:
            continue
        nid = arr[0]
        nid_title_description_dict[nid] = {}
        if len(arr) >= 2:
            nid_title_description_dict[nid]["title"] = arr[1]
        if len(arr) >= 3:
            nid_title_description_dict[nid]["description"] = arr[2]
        if len(arr) >= 4:
            nid_title_description_dict[nid]["gk_tags"] = arr[3:]

def thread_write_dict(nid, title, description, gk_tags):
    """
    线程锁，写入字典
    """
    global nid_title_description_dict
    global lock_dict
    global lock_print
    global is_video
    # if is_video == True:
    if title or description or gk_tags:
        lock_dict.acquire()
        nid_title_description_dict[nid] = {}
        nid_title_description_dict[nid]["title"] = title
        nid_title_description_dict[nid]["description"] = description
        nid_title_description_dict[nid]["gk_tags"] = gk_tags
        if type(gk_tags) == list:
            thread_print_content([nid, title, description] + gk_tags)
        lock_dict.release()
    else:
        return

def thread_print_content(list_content):
    """
    线程锁，print的锁
    """
    global lock_print
    lock_print.acquire()
    print '\t'.join(list_content)
    lock_print.release()


def get_host_list(bns):
    """Get host_list.
    """
    cmd = "get_instance_by_service " + bns + " -ip"
    res = os.popen(cmd).read().strip()   # 多线程在使用os.popen会报错
    host_list = res.split("\n")
    return host_list

def multi_thread_work(nid_list, start, end, host, port):
    """
    多线程
    """
    global nid_title_description_dict
    global is_video
    processed_nid_list = nid_list[start:end + 1]
    for line in processed_nid_list:
        nid_cnt = line.strip().split()
        nid = nid_cnt[0]
        cnt = int(nid_cnt[1])
        if cnt < 2:
            continue
        if nid not in nid_title_description_dict:
            title, description, gk_tags = "", "", []
            slots = tool_nid_to_nid_info.get_nid_info(nid, host, port)
            xxxxx# 省略
            thread_write_dict(nid, title, description, gk_tags)
    return

nid_poi_file_path = "../" + folder + "/nid_poi_file"
with open(nid_poi_file_path, "r") as file:
    line = file.readlines()


host_list = get_host_list(bns)
# 分100个线程
thread_q = []
part_len = 1 + (len(line) // 100)
for i in range(100):
    rand_host_item = random.choice(host_list)
    rand_host_port = rand_host_item.split(" ")
    if len(rand_host_port) == 3:
        host, port = rand_host_port[0], rand_host_port[2]
    p=Thread(target=multi_thread_work, args=(line, i * part_len, -1 + \
        (i + 1) * part_len, host, port))
    p.setDaemon(True)
    p.start()
    thread_q.append(p)
for p in thread_q:
    p.join()    

