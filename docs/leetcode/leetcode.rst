.. knowledge_record documentation master file, created by
   sphinx-quickstart on Tue July 4 21:15:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

******************
leetcode
******************

二分查找类
==================

二分查找
---------------------

二分查找::
def binary_search(target, array):
	l = 0
	r = len(array)-1
	while l<=r:
		mid = (l+r)//2
		if array[mid]==target:
			return mid
		elif array[mid]<target:
			l = mid + 1
		else:
			r = mid – 1
	return False





