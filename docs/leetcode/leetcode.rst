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
	:width: 400


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


	
	

排序
====================


快排
-------------------
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
				if not root:
					return None
				res.append(root.val)
				helper(root.left)
				helper(root.right)
			helper(root)
			return res
		
迭代::

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

注意点：

1.为什么这里要用stack 而不是 queue：
| 因为这是深度优先，DFS。stack的话就是先处理子节点，深入到底然后再往上的根。

2. 特别注意由于这里是stack，所以前序遍历的时候先stack.append(node.right)

中序遍历
---------------------
递归::

	class Solution:
		def inorderTraversal(self, root: TreeNode) -> List[int]:
			res = []
			def helper(root):
				if not root:
					return None
				helper(root.left)
				res.append(root.val)
				helper(root.right)
			helper(root)
			return res

迭代::

	class Solution:
		def inorderTraversal(self, root: TreeNode) -> List[int]:
			res = []
			if not root:
				return res
			stack = []
			while root or stack:
				while root:
					stack.append(root)
					root = root.left
				root = stack.pop()
				res.append(root.val)
				root = root.right
			return res

后续遍历
----------------------
递归::

	class Solution:
		def postorderTraversal(self, root: TreeNode) -> List[int]:
			res = []
			def helper(root):
				if not root:
					return None
				helper(root.left)
				helper(root.right)
				res.append(root.val)
			helper(root)
			return res

迭代::

	class Solution:
		def postorderTraversal(self, root: TreeNode) -> List[int]:
			res = []
			if not root:
				return res
			stack = [root]
			while stack:
				node = stack.pop()
				if node.left:
					stack.append(node.left)
				if node.right:
					stack.append(node.right)
				res.append(node.val)
			return res[::-1]

注意点：

后序遍历是 左右中，然后我们使用了stack，所以录入的时候是左右中，（先进后出），然后对结果[::-1] 取逆序就好了。 [::-1]这个操作对 string和list 都适用的


层次遍历
-----------------------

leetcode 102. 二叉树的层次遍历::

	class Solution:
		def levelOrder(self, root: TreeNode) -> List[List[int]]:
			if not root:
				return []
			cur_level, res = [root], []
			while cur_level:
				temp = []
				next_level = []
				for node in cur_level:
					temp.append(node.val)
					if node.left:
						next_level.append(node.left)
					if node.right:
						next_level.append(node.right)
				res.append(temp)
				cur_level = next_level
			return res


相同的树
----------------
leetcode 100. 

给定两个二叉树，编写一个函数来检验它们是否相同。

如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。::

	# Definition for a binary tree node.
	# class TreeNode:
	#     def __init__(self, x):
	#         self.val = x
	#         self.left = None
	#         self.right = None

	class Solution:
		def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
			if (p==None and q==None):
				return True
			if p==None or q == None:
				return False
			if p.val!= q.val:
				return False
			return self.isSameTree(p.left,q.left) and self.isSameTree(p.right,q.right)


树的子结构
----------------

剑指 Offer 26. 

输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

B是A的子结构， 即 A中有出现和B相同的结构和节点值。::

	# Definition for a binary tree node.
	# class TreeNode:
	#     def __init__(self, x):
	#         self.val = x
	#         self.left = None
	#         self.right = None

	class Solution:
		def judge(self,a,b):
			if not b:
				return True
			if not a:
				return False
			if a.val!= b.val:
				return False
			return self.judge(a.left,b.left) and self.judge(a.right,b.right)

		def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
			if (B==None or A==None):
				return False
			if self.judge(A,B):
				return True
			return self.isSubStructure(A.left,B) or self.isSubStructure(A.right,B)


我的题解

https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/solution/chao-hao-dong-ke-fu-yong-tong-guo-issametreena-dao/

| 解题思路
| 因为刚刚做完 leetcode第100题----isSameTree ： https://leetcode-cn.com/problems/same-tree/
| 所以合理的衍生一下，非常的好理解。

| 最开始的想法是：我们对A中的结点去遍历，每个结点都调用之前写的 isSameTree，如果A中的某个结点和B完全一样，那不就找到了吗！
| 后来发现有个bug，就是 B不仅可以是 A的末端，也可以是中间的某段。（A可以比B 多一点分叉）
| 所以只要把isSameTree的条件放宽一点就好了：不需要完全相等，只要在B的所有结点内都相等就好了。
| isSameTree函数 放宽条件，改写成本文中的judge函数。

| 第一个judge函数是判断，第二个就是不断的去调用。

| 作者：luock
| 链接：https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/solution/chao-hao-dong-ke-fu-yong-tong-guo-issametreena-dao/
| 来源：力扣（LeetCode）
| 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

平衡二叉树
---------------
剑指 Offer 55 - II. 

输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。::

	# Definition for a binary tree node.
	# class TreeNode:
	#     def __init__(self, x):
	#         self.val = x
	#         self.left = None
	#         self.right = None

	class Solution:
		def isBalanced(self, root: TreeNode) -> bool:
			def helper(root):
				if not root:
					return 0
				left = helper(root.left)
				if left == -1:
					return -1
				right = helper(root.right)
				if right ==-1:
					return -1
				if abs(left-right)>1:
					return -1
				else:
					return max(left,right)+1
			depth = helper(root)
			if depth ==-1:
				return False
			else:
				return True


对称的二叉树
-----------------

剑指 Offer 28. 

请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的::

	# Definition for a binary tree node.
	# class TreeNode:
	#     def __init__(self, x):
	#         self.val = x
	#         self.left = None
	#         self.right = None

	class Solution:
		def check_list(self,list_level):
			lens = len(list_level)
			# if lens%2 !=0:
			#     return False
			for i in range(0,lens//2):
				if list_level[i]!=list_level[lens-i-1]:
					return False
			return True

		def isSymmetric(self, root: TreeNode) -> bool:
			if not root:
				return True
			this_level = [root]
			while this_level:
				temp = []
				next_level = []
				for node in this_level:
					if not node:
						temp.append(None)
					else:
						temp.append(node.val)
						next_level.append(node.left)
						next_level.append(node.right)
				if self.check_list(temp)==False:
					return False
				this_level = next_level
			return True


二叉树的镜像	
-------------------		
剑指 Offer 27.

请完成一个函数，输入一个二叉树，该函数输出它的镜像。::

	# Definition for a binary tree node.
	# class TreeNode:
	#     def __init__(self, x):
	#         self.val = x
	#         self.left = None
	#         self.right = None

	class Solution:
		def mirrorTree(self, root: TreeNode) -> TreeNode:
			'''
			递归
			'''
			# if not root:
			#     return None
			# root.left,root.right = self.mirrorTree(root.right),self.mirrorTree(root.left)
			# return root
			'''
			迭代
			'''
			if not root:
				return None
			queue = [root]
			while queue:
				node = queue.pop(0)
				if node:
					node.left,node.right = node.right, node.left
					queue.append(node.left)
					queue.append(node.right)
			return root


从前序与中序遍历序列构造二叉树
----------------------------------------

leetcode 105. 

根据一棵树的前序遍历与中序遍历构造二叉树。

注意:

你可以假设树中没有重复的元素。::

	# Definition for a binary tree node.
	# class TreeNode:
	#     def __init__(self, x):
	#         self.val = x
	#         self.left = None
	#         self.right = None

	class Solution:
		def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
			# if not (preorder and inorder):
			#     return None
			# root = TreeNode(preorder[0])
			# mid_idx = inorder.index(preorder[0])
			# root.left = self.buildTree(preorder[1:mid_idx+1],inorder[:mid_idx])
			# root.right = self.buildTree(preorder[mid_idx+1:],inorder[mid_idx+1:])
			# return root
			def building(preorder,inorder):
				if not (preorder and inorder):
					print(preorder)
					return None
				root_val = preorder[0]
				root = TreeNode(root_val)
				root_index = inorder.index(root_val)

				root.left = building(preorder[1:root_index+1],inorder[:root_index])
				root.right = building(preorder[root_index+1:],inorder[root_index+1:])
				return root
			return building(preorder,inorder)

从中序与后序遍历序列构造二叉树
--------------------------------------

leetcode 106. 

根据一棵树的中序遍历与后序遍历构造二叉树。

注意:

你可以假设树中没有重复的元素。::

	# Definition for a binary tree node.
	# class TreeNode:
	#     def __init__(self, x):
	#         self.val = x
	#         self.left = None
	#         self.right = None

	class Solution:
		def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
			if not (inorder and postorder):
				return None
			root_val = postorder[-1]
			root = TreeNode(root_val)
			root_index = inorder.index(root_val)
			lens = len(inorder)
			root.right = self.buildTree(inorder[root_index+1:],postorder[root_index:-1])
			root.left = self.buildTree(inorder[:root_index],postorder[:root_index])
			return root


动态规划
===================

最长回文子串
-------------------

leetcode 5. 

给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。::

	class Solution:
		def longestPalindrome(self, s: str) -> str:
			def check(string,index):
				i=0
				while index-i>=0 and index+i<=len(string)-1:
					if string[index-i]==string[index+i]:
						i+=1
					else:
						return i-1
				return i-1
			res = []
			if len(s)<=1:
				return s
			for i in range(len(s)):
				temp = check(s,i)
				if 2*temp +1>len(res):
					res = s[i-temp:i]+s[i:i+temp+1]
				temp = check(s[:i]+'#'+s[i:],i)
				if 2*temp +1>len(res):
					res = s[i-temp:i]+s[i:i+temp]
			return res
