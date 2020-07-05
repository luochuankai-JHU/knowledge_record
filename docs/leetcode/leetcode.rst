.. knowledge_record documentation master file, created by
   sphinx-quickstart on Tue July 4 21:15:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

******************
Leetcode
******************


**面试写题的时候可以把注释也写上**

**写题的时候，在最前面写几个例子**


二分查找类
==================



二分查找
--------------
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


搜索旋转排序数组
------------------------------------
leetcode 33. 

假设按照升序排序的数组在预先未知的某个点上进行了旋转。

( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。

搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。

你可以假设数组中不存在重复的元素。

你的算法时间复杂度必须是 O(log n) 级别。::

	class Solution:
		def search(self, nums: List[int], target: int) -> int:
			def binary_search(List, target):
				l, r = 0, len(List) - 1
				while l <= r:
					mid = (l + r) // 2
					if List[mid] == target:
						return mid
					elif List[mid] > target:
						r = mid - 1
					elif List[mid] < target:
						l = mid + 1
				return -1
			
			l,r = 0,len(nums)-1
			while l<=r:
				mid = (l+r)//2
				if nums[mid]>=nums[l]:
					# 左边有序
					if nums[l]<=target<=nums[mid]:
						return binary_search(nums[l:mid+1], target)+l if binary_search(nums[l:mid+1], target)!=-1 else -1
					else:
						l = mid+1
				elif nums[mid]<=nums[r]:
					# 右边有序
					if nums[mid]<=target<=nums[r]:
						return binary_search(nums[mid:r+1], target)+mid if binary_search(nums[mid:r+1], target)!=-1 else -1
					else:
						r = mid-1
			return -1


在排序数组中查找元素的第一个和最后一个位置
---------------------------------------------------------
leetcode 34. 

给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。

你的算法时间复杂度必须是 O(log n) 级别。

如果数组中不存在目标值，返回 [-1, -1]。::

	class Solution:
		def searchRange(self, nums: List[int], target: int) -> List[int]:
			def get_left(nums,target):
				l,r = 0,len(nums)-1
				res = -1
				while l<=r:
					mid = (l+r)//2
					if nums[mid]==target:
						res = mid
						r = mid - 1
						if mid==0:
							return res
					elif  nums[mid]<target:
						l = mid + 1
					elif nums[mid]>target:
						r = mid - 1
				return res

			def get_right(nums,target):
				l,r = 0,len(nums)-1
				find = 0
				res = -1
				while l<=r:
					mid = (l+r)//2
					if nums[mid]==target:
						res = mid
						l = mid + 1
						if mid==len(nums)-1:
							return res
					elif  nums[mid]<target:
						l = mid + 1
					elif nums[mid]>target:
						r = mid - 1
				return res

			left = get_left(nums,target)
			if left==-1:
				return [-1,-1]
			right = get_right(nums,target)
			return [left,right]




搜索插入位置
-------------------------------

leetcode 35. 

给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

你可以假设数组中无重复元素。::

	class Solution:
		def searchInsert(self, nums: List[int], target: int) -> int:
			l,r = 0, len(nums)-1
			while l<=r:
				mid = (l+r)//2
				if nums[mid]==target:
					return mid
				elif nums[mid]>target:
					r = mid - 1
				else:
					l = mid + 1
			return l









寻找旋转排序数组中的最小值
--------------------------------------------
leetcode 153. 

假设按照升序排序的数组在预先未知的某个点上进行了旋转。

( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。

请找出其中最小的元素。

你可以假设数组中不存在重复元素。::

	class Solution:
		def findMin(self, nums: List[int]) -> int:
			l, r = 0, len(nums) - 1
			while l<=r:
				mid = (l+r)//2
				if nums[mid]>nums[r]:
					l = mid + 1
				elif nums[mid]<nums[r]:
					r = mid
				if l == r-1 or l==r:
					return min(nums[l], nums[r])


搜索旋转排序数组 II
----------------------------------
leetcode 81. 


.. image:: ../../_static/leetcode/81.png
	:align: center


假设按照升序排序的数组在预先未知的某个点上进行了旋转。

( 例如，数组 [0,0,1,2,2,5,6] 可能变为 [2,5,6,0,0,1,2] )。

编写一个函数来判断给定的目标值是否存在于数组中。若存在返回 true，否则返回 false。::

	class Solution:
		def search(self, nums: List[int], target: int) -> bool:
			def binary_search(nums,target):
				l, r = 0, len(nums) - 1
				while l <= r:
					mid = (l+r) // 2
					if nums[mid] == target:
						return True
					elif nums[mid] < target:
						l = mid + 1
					elif nums[mid] > target:
						r = mid -1 
				return False
			
			l, r = 0, len(nums) - 1
			while l <= r:
				mid = (l+r) // 2
				if target in [nums[mid],nums[r],nums[l]]:
					return True
				if nums[r] == nums[l]:
					l = l + 1
					r = r - 1
					continue 
				if nums[mid] <= nums[r]:
					# 右边有序
					if nums[mid] < target < nums[r]:
						return binary_search(nums[mid:r],target)
					else:
						r = mid -1
				else:
					# 左边有序
					if nums[l] < target < nums[mid]:
						return binary_search(nums[l:mid],target)
					else:
						l = mid + 1
			return False


	
	

快排
====================
https://www.cnblogs.com/Jinghe-Zhang/p/8986585.html

快排::

	def parttion(v, left, right):
		key = v[left]
		low = left
		high = right
		while low < high:
			while (low < high) and (v[high] >= key):
				high -= 1
			v[low] = v[high]
			while (low < high) and (v[low] <= key):
				low += 1
			v[high] = v[low]
			v[low] = key
		return low
	def quicksort(v, left, right):
		if left < right:
			p = parttion(v, left, right)
			quicksort(v, left, p-1)
			quicksort(v, p+1, right)
		return v

	s = [6, 8, 1, 4, 3, 9, 5, 4, 11, 2, 2, 15, 6]
	print("before sort:",s)
	s1 = quicksort(s, left = 0, right = len(s) - 1)
	print("after sort:",s1)


树的遍历：
======================

https://leetcode-cn.com/problems/binary-tree-preorder-traversal/solution/di-gui-he-die-dai-by-powcai-5/


前序遍历
---------------

递归::

	class Solution(object):
		def preorderTraversal(self, root):
			"""
			:type root: TreeNode
			:rtype: List[int]
			"""
			res = []
			def helper(root):
				if not root:return 
				res.append(root.val)
				helper(root.left)
				helper(root.right)
			helper(root)
			return res
		
迭代::


class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        p = root
        stack = []
        while p or stack:
            while p:
                res.append(p.val)
                stack.append(p)
                p = p.left
            p = stack.pop().right
        return res

中序遍历
---------------------

后续遍历
----------------------

层次遍历
-----------------------

		
二叉树的前序,中序,后序,层序遍历的递归和迭代,一起打包送个你们!嘻嘻

144. 二叉树的前序遍历

思路:

递归:就是依次输出根,左,右,递归下去

迭代:使用栈来完成,我们先将根节点放入栈中,然后将其弹出,依次将该弹出的节点的右节点,和左节点,**注意顺序,**是右,左,为什么?因为栈是先入后出的,我们要先输出右节点,所以让它先进栈.

代码:

递归:


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        def helper(root):
            if not root:
                return 
            res.append(root.val)
            helper(root.left)
            helper(root.right)
        helper(root)
        return res
迭代:


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        if not root:
            return res
        stack = [root]
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res
145. 二叉树的后序遍历

思路:

递归:同理,顺序:左,右,根

迭代:这就很上面的先序一样,我们可以改变入栈的顺序,刚才先序是从右到左,我们这次从左到右,最后得到的结果取逆.

代码:

递归:


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        def helper(root):
            if not root:
                return 
            helper(root.left)
            helper(root.right)
            res.append(root.val)
        helper(root)
        return res
迭代:


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        if not root:
            return res
        stack = [root]
        while stack:
            node = stack.pop()
            if node.left :
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
            res.append(node.val)
        return res[::-1]
94. 二叉树的中序遍历

思路:

递归:顺序,左右根

非递归:这次我们用一个指针模拟过程

代码:

递归:


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        def helper(root):
            if not root:
                return 
            helper(root.left)
            res.append(root.val)
            helper(root.right)
        helper(root)
        return res
迭代:


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        if not root:
            return res
        stack = []
        cur = root
        while stack or cur:
            while cur:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            res.append(cur.val)
            cur = cur.right
        return res
102. 二叉树的层次遍历

思路:

非常典型的BFS

代码:


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []

        res,cur_level = [],[root]
        while cur_level:
            temp = []
            next_level = []
            for i in cur_level:
                temp.append(i.val)

                if i.left:
                    next_level.append(i.left)
                if i.right:
                    next_level.append(i.right)
            res.append(temp)
            cur_level = next_level
        return res


回文
================



