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


或者在第二个函数用一下伪层次遍历::

    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.left = None
    #         self.right = None
    class Solution:
        def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
            def judge(a,b):
                if not b:
                    return True
                if not a:
                    return False
                if a.val!= b.val:
                    return False
                return judge(a.left,b.left) and judge(a.right,b.right)

            if (A==None or B==None):
                return False
            queue = [A]
            while queue:
                node = queue.pop(0)
                if judge(node,B):
                    return True             
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)  
            return False 


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
                if temp!=temp[::-1]:
                    return False
                this_level = next_level
            return True




二叉树中和为某一值的路径**好题**
---------------------------------
剑指 Offer 34. 

**好题目！！！**

输入一棵二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。从树的根节点开始往下一直到叶节点所经过的节点形成一条路径。
            
.. image:: ../../_static/leetcode/剑指34.png
    :align: center
    :width: 400

::
    def pathSum(self, root: TreeNode, target: int) -> List[List[int]]:
        res, path = [], []
        def order(root):
            if not root:
                return None
            path.append(root.val)
            if sum(path)==target and not root.right and not root.left:
                res.append(path[:])
            order(root.left)
            order(root.right)
            path.pop()
        order(root)
        return res
    
注意！res.append(path[:]) 这里一定要是 path[:]，因为list是可变变量，直接append是浅拷贝，最后res里面只会留下空数组


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

股票的最大利润
------------------------------
剑指 Offer 63. 

.. image:: ../../_static/leetcode/剑指63.png
    :align: center
    :width: 400
    
假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？::

    def maxProfit(self, prices: List[int]) -> int:
        if len(prices)<=0:
            return 0
        Max,Min = prices[0],prices[0]
        res = 0
        for i in range(len(prices)):
            if prices[i]>Max:
                Max = prices[i]
                temp = Max-Min
                res = max(temp,res)
            elif  prices[i]<Min:
                Min = prices[i]
                Max = prices[i]                 
        return res


顺时针打印矩阵
------------------------
剑指 Offer 29. 

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

示例 1：

| 输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
| 输出：[1,2,3,6,9,8,7,4,5]

示例 2：

| 输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
| 输出：[1,2,3,4,8,12,11,10,9,5,6,7]

一种很憨憨的解法，一板一眼的去做::
    class Solution:
        def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
            res = []
            def turn_right(matrix,res):
                res+=matrix[0]
                matrix = matrix[1:]
                return matrix, res

            def turn_down(matrix,res):
                new_matrix = []
                for line in matrix:
                    res.append(line[-1])
                    line = line[:-1]
                    new_matrix.append(line)
                return new_matrix,res

            def turn_left(matrix,res):
                res+=matrix[-1][::-1]
                matrix = matrix[:-1]
                return matrix, res
            
            def turn_up(matrix,res):
                new_matrix = []
                temp = []
                for line in matrix:
                    temp.append(line[0])
                    line = line[1:]
                    new_matrix.append(line)
                res += temp[::-1]
                return new_matrix,res
            i = 0
            while len(matrix)>0 and len(matrix[0])>0:
                if i%4==0:
                    matrix,res = turn_right(matrix,res)
                    i+=1
                    continue
                if i%4==1:
                    matrix,res = turn_down(matrix,res)
                    i+=1
                    continue
                if i%4==2:
                    matrix,res = turn_left(matrix,res)
                    i+=1
                    continue
                if i%4==3:
                    matrix,res = turn_up(matrix,res)
                    i+=1
                    continue
            return res


字符串的排列
--------------------
剑指 Offer 38. 

输入一个字符串，打印出该字符串中字符的所有排列。

你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。

示例:

| 输入：s = "abc"
| 输出：["abc","acb","bac","bca","cab","cba"]

我的一个憨憨解法
::
    def permutation(self, s: str) -> List[str]:
        def insert(res,char):
            temp = []
            for string in res:
                for i in range(len(string)+1):
                    temp.append(string[:i]+char+string[i:])
            temp = list(set(temp))
            return temp

        if len(s)==0:
            return []
        res = [s[0]]
        for i in range(1,len(s)):
            res = insert(res,s[i])
        return res

从第一个字符开始维护一个list，里面的内容是答案。然后每次都全部插入，再去重。如果不让用set去重可以字典啊或者直接set.add

想法很朴素，写起来也很朴素，但是时间和空间使用率接近双百分


数组中出现次数超过一半的数字
-------------------------------------
剑指 Offer 39. 

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

示例 1:

| 输入: [1, 2, 3, 2, 2, 2, 5, 4, 2]
| 输出: 2
::
    def majorityElement(self, nums: List[int]) -> int:
        if nums==[]:
            return []
        count = 1
        res = nums[0]
        for i in range(1,len(nums)):
            if nums[i]==res:
                count+=1
            else:
                count -=1
                if count ==0:
                    res = nums[i]
                    count = 1
        return res
        
一个漂亮的解法。维护一个res和count。如果当前遍历到的数和res相等，count就+1，不不然就-1。减到0 res就换人。 记得换人后把count重新设为1 !!!

链表
===================

链表中倒数第k个节点
------------------------

剑指 Offer 22. 

输入一个链表，输出该链表中倒数第k个节点。

为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。

例如，一个链表有6个节点，从头节点开始，它们的值依次是1、2、3、4、5、6。这个链表的倒数第3个节点是值为4的节点。::

    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        l, r = head, head
        i = 0
        while i<k and r:
            if not r:
                return False
            r = r.next
            i+=1

        while r:
            r = r.next
            l = l.next
            
        return l

明显的双指针题目


反转链表
------------------
剑指 Offer 24. 

定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。

示例:

输入: 1->2->3->4->5->NULL

输出: 5->4->3->2->1->NULL

::

    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        pre = None
        cur = head
        # 遍历链表，while循环里面的内容其实可以写成一行
        # 这里只做演示，就不搞那么骚气的写法了
        while cur:
            # 记录当前节点的下一个节点
            tmp = cur.next
            # 然后将当前节点指向pre
            cur.next = pre
            # pre和cur节点都前进一位
            pre = cur
            cur = tmp
        return pre    

https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/solution/dong-hua-yan-shi-duo-chong-jie-fa-206-fan-zhuan-li/

.. image:: ../../_static/leetcode/剑指24.png
    :align: center
    :width: 200
    
合并两个排序的链表
-------------------------
    
剑指 Offer 25. 

输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。

示例1：

输入：1->2->4, 1->3->4

输出：1->1->2->3->4->4

::
    # class ListNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.next = None

    class Solution:
        def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
            res = temp = ListNode(0)
            while l1 and l2:
                if l1.val>=l2.val:
                    temp.next = l2
                    l2 = l2.next
                else:
                    temp.next = l1
                    l1 = l1.next
                temp = temp.next
            if l1:
                temp.next = l1
            if l2:
                temp.next = l2
            return res.next


注意： temp = temp.next 这句话千万不能忘，然后开头的res = temp = ListNode(0) 也很关键！


另外：不用额外空间合并两个排序的list

不用额外空间合并两个排序的list
---------------------------------
::
    list1 = [1,3,5,7,8,9,13]
    list2 = [0,3,5,8,13,16]

    i,j = 0,0
    while i<=len(list1)-1 and list2:
        print(i)
        if list2[0]<=list1[i]:
            num = list2.pop(0)
            list1.insert(i,num)
        else:
            i+=1
    if list2:
        list1+=list2