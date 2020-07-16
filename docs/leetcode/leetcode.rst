.. knowledge_record documentation master file, created by
   sphinx-quickstart on Tue July 4 21:15:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

******************
Leetcode
******************

这里保存一些我做题的解答和心得

**tips**


面试写题的时候可以把注释也写上

写题的时候，在最前面写几个例子


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


emmmm 上面这样写好蠢啊

剑指53跟这个几乎一样
::

	def search(self, nums: List[int], target: int) -> int:

        def get_first(nums,target):
            l, r = 0, len(nums)-1
            while l <= r:
                mid = (l + r)//2
                if nums[mid]>=target:
                    r = mid -1
                elif nums[mid] < target:
                    l = mid + 1
            return l
        
        def get_last(nums,target):
            l, r = 0, len(nums)-1
            while l <= r:
                mid = (l + r)//2
                if nums[mid] <= target:
                    l = mid + 1
                elif nums[mid] > target:
                    r = mid - 1
            return r 
        
        r = get_last(nums,target)
        l = get_first(nums,target)
        if r < l:
            return 0
        else:
            return r - l +1


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


0～n-1中缺失的数字
--------------------------
剑指 Offer 53 - II. 

| 一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。
| 在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。

| 示例 1:
| 输入: [0,1,3]
| 输出: 2

| 示例 2:
| 输入: [0,1,2,3,4,5,6,7,9]
| 输出: 8    
    
::

    def missingNumber(self, nums: List[int]) -> int:
        i, j = 0, len(nums) - 1
        while i <= j:
            m = (i + j) // 2
            if nums[m] == m: i = m + 1
            else: j = m - 1
        return i

别人的解法还是很简洁的

相比之下，我的解法有些冗余::

    def missingNumber(self, nums: List[int]) -> int:
        l, r = 0, len(nums)
        if nums[0] != 0:
            return 0
        if nums[-1] != len(nums):
            return len(nums)
        while l <= r:
            mid = (l + r) // 2
            if mid == nums[mid]:
                l = mid
            else:
                r = mid
            if r == l + 1:
                return (nums[r] + nums[l])//2
				
| 想法其实很简单，就二分查找。因为这个题有个限定，是左边从0开始，所以最开始要讨论一下缺失两边的情况。
| 然后中间的时候直接用if mid == nums[mid] 就可以了。

| 有个想法。是不是 l = mid 这种地方，要不就都用 mid +1 ， mid-1 要不就都不加都不减。不然容易出问题
| 反正最后那个if r == l + 1: return (nums[r] + nums[l])//2 直接耍流氓很舒服

| 还是多多学习别人的吧！ 巧妙的利用了 二分查找之后，导致while停止循环的情况一定是： r在查找值的左边，l在查找值的右边。 

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

数组中的逆序对
----------------------
剑指 Offer 51. 

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。

| 输入: [7,5,6,4]
| 输出: 5

这个是真不会...做法是用到归并排序...


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


二叉树中和为某一值的路径
---------------------------------
剑指 Offer 34. 

**好题目！！！**

.. image:: ../../_static/leetcode/剑指34.png
    :align: center
    :width: 400
    
输入一棵二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。从树的根节点开始往下一直到叶节点所经过的节点形成一条路径。::
            
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


连续子数组的最大和
-------------------------
剑指 Offer 42. 

输入一个整型数组，数组里有正数也有负数。数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

要求时间复杂度为O(n)。

示例1:

| 输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
| 输出: 6
| 解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。

::

    def maxSubArray(self, nums: List[int]) -> int:
        if len(nums)==0:
            return 0
        res, temp = nums[0], nums[0]
        for i in range(1,len(nums)):
            temp = max(nums[i],temp+nums[i])
            res = max(temp,res)
        return res

值得再去好好想想


更进一步，请看下一题：

乘积最大子数组
------------------------

leetcode 152. 

给你一个整数数组 nums ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

示例 1:

| 输入: [2,3,-2,4]
| 输出: 6
| 解释: 子数组 [2,3] 有最大乘积 6。
| 示例 2:

| 输入: [-2,0,-1]
| 输出: 0
| 解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。

::

    def maxProduct(self, nums: List[int]) -> int:
        if not nums: return 
        res = nums[0]
        pre_max = nums[0]
        pre_min = nums[0]
        for num in nums[1:]:
            cur_max = max(pre_max * num, pre_min * num, num)
            cur_min = min(pre_max * num, pre_min * num, num)
            res = max(res, cur_max)
            pre_max = cur_max
            pre_min = cur_min
        return res


链接：https://leetcode-cn.com/problems/maximum-product-subarray/solution/duo-chong-si-lu-qiu-jie-by-powcai-3/

| 思路很巧妙！ 因为这个题目比上一题难在，虽然现在的cur可能是一个很小的负数（但是绝对值大），再乘一个负数后就会变得很大。所以绝对值很重要。
| 大正数和小负数（绝对值大）都要保存记录。而不是像上一题只用记录一个就行


还有一种解法暂时没太明白，也先记录下来。

思路三：根据符号的个数 [^2]

| 当负数个数为偶数时候，全部相乘一定最大
| 当负数个数为奇数时候，它的左右两边的负数个数一定为偶数，只需求两边最大值
| 当有 0 情况，重置就可以了

::

    def maxProduct(self, nums: List[int]) -> int:
        reverse_nums = nums[::-1]
        for i in range(1, len(nums)):
            nums[i] *= nums[i - 1] or 1
            reverse_nums[i] *= reverse_nums[i - 1] or 1
        return max(nums + reverse_nums)

把数组排成最小的数
------------------------
剑指 Offer 45. 

输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。

示例 1:

| 输入: [10,2]
| 输出: "102"
| 示例 2:

| 输入: [3,30,34,5,9]
| 输出: "3033459"

::

    def minNumber(self, nums: List[int]) -> str:
        if nums==[]:
            return ''
        nums = [str(x) for x in nums]
        for i in range(0,len(nums)-1):
            for j in range(i+1,len(nums)):
                if int(nums[i] + nums[j] > nums[j] + nums[i]):
                    nums[i], nums[j] = nums[j], nums[i]
        return ''.join(nums)
        
O(n2)的解法，类似冒泡排序。

有一种O(nlogn)的解法，类似于快排。暂时不理解，先记录下来：

https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/solution/mian-shi-ti-45-ba-shu-zu-pai-cheng-zui-xiao-de-s-4/

::

    def minNumber(self, nums: List[int]) -> str:
        def fast_sort(l , r):
            if l >= r: return
            i, j = l, r
            while i < j:
                while strs[j] + strs[l] >= strs[l] + strs[j] and i < j: j -= 1
                while strs[i] + strs[l] <= strs[l] + strs[i] and i < j: i += 1
                strs[i], strs[j] = strs[j], strs[i]
            strs[i], strs[l] = strs[l], strs[i]
            fast_sort(l, i - 1)
            fast_sort(i + 1, r)
        
        strs = [str(num) for num in nums]
        fast_sort(0, len(strs) - 1)
        return ''.join(strs)

里面涉及到一些数学推导与证明，评论区和下面其他大佬的解答里面有证明。


和为s的连续正数序列
-----------------------------
剑指 Offer 57 - II. 

输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。

序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。

| 示例 1：
| 输入：target = 9
| 输出：[[2,3,4],[4,5]]

| 示例 2：
| 输入：target = 15
| 输出：[[1,2,3,4,5],[4,5,6],[7,8]]
::

    def findContinuousSequence(self, target: int) -> List[List[int]]:
        if target<=2:
            return None
        l,r = 1,1
        res = []
        the_sum = 1
        while l<=target//2:
            if the_sum<target:
                r+=1
                the_sum+=r
            elif the_sum>target:
                the_sum-=l 
                l+=1
            elif the_sum==target:
                res.append([x for x in range(l,r+1)])
                the_sum-=l 
                l+=1
        return res

经典双指针题目



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
        
        
礼物的最大价值
----------------------
剑指 Offer 47. 

| 在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。
| 你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。
| 给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？


示例 1:

输入: 
| [
|   [1,3,1],
|   [1,5,1],
|   [4,2,1]
| ]
| 输出: 12
| 解释: 路径 1→3→5→2→1 可以拿到最多价值的礼物

::

    def maxValue(self, grid: List[List[int]]) -> int:
        if grid==[]:
            return 0
        for j in range(len(grid)):
            for i in range(len(grid[0])):
                if i==0 and j==0:
                    continue
                if j==0 and i!=0:
                    grid[j][i] += grid[j][i-1]
                if i==0 and j!=0:
                    grid[j][i] += grid[j-1][i]
                if i!=0 and j!=0:
                    grid[j][i] += max(grid[j-1][i],grid[j][i-1])
        return grid[-1][-1]

注意，最后一个if（讨论中间的格子），不要写else.....血的教训。依然是if，不然会和第三个if 组成if...else。

除了第一行和第一列，其他的情况： 选择 max（左边，上面）+ 自己那一格

更方便的做法是在左边和上面都补上一列0，这样就不用分四种情况讨论了，公式能通用。

https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof/solution/mian-shi-ti-47-li-wu-de-zui-da-jie-zhi-dong-tai-gu/

最长不含重复字符的子字符串
---------------------------------
剑指 Offer 48. 

请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。

示例 1:

| 输入: "abcabcbb"
| 输出: 3 
| 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
| 示例 2:

| 输入: "bbbbb"
| 输出: 1
| 解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
| 示例 3:

| 输入: "pwwkew"
| 输出: 3
| 解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。

::

    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s)<=1:
            return len(s)
        i = 0
        res = 1
        for j in range(1,len(s)):
            if s[j] not in s[i:j]:
                pass
            else:
                i = s[i:j].index(s[j]) + i + 1
            res = max(res,j-i+1)
        return res


丑数
--------------
剑指 Offer 49. 

我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。

| 示例:
| 输入: n = 10
| 输出: 12
| 解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。

::

    def nthUglyNumber(self, n: int) -> int:
        index= 1
        ugly = [1]
        dp2,dp3,dp5 = 0,0,0
        while index <= n-1:
            cur = min(2*ugly[dp2], 3*ugly[dp3], 5*ugly[dp5])
            if cur == 2*ugly[dp2]:
                dp2 += 1
            if cur == 3*ugly[dp3]:
                dp3 += 1
            if cur == 5*ugly[dp5]:
                dp5 += 1
            index += 1
            ugly.append(cur)
        return ugly[-1]

| 最朴素（暴力）的解法是这样：
| 首先我们明白，类比跳台阶那个题目，任意一个新的丑数，一定是之前的丑数 *2 或 *3 或 *5 得来的。
| 那么最暴力的做法就是，要生成一个新的丑数，把之前所有的元素都乘 2，3，5。然后找到最小的那个（注意！不能只选倒数三个人，因为10=2*5）

| 这里造成冗余的原因是：
| 很多之前的数已经没有意义了，比如3，如果已经通过3*2得到了6，那么下次就不需要再算3*2了。

| 由此，这道题可以维护三个指针。

| 注意，这里用三个if的原因是为了解决这个难题：得到6的时候，不仅是2*3，其实也是3*2。所以这两种可能性都要失效，所以这两个指针都要+1

| 再要注意的地方是，我最开始写的是while index<= n。这样算的是第n+1个丑数


Z 字形变换
-----------------
leetcode 6. 

将一个给定字符串根据给定的行数，以从上往下、从左到右进行 Z 字形排列。

.. image:: ../../_static/leetcode/6.png
	:align: center
	:width: 400
	
	
圆圈中最后剩下的数字
----------------------------
剑指 Offer 62. 

0,1,,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字。求出这个圆圈里剩下的最后一个数字。

例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。

| 示例 1：
| 输入: n = 5, m = 3
| 输出: 3

| 示例 2：
| 输入: n = 10, m = 17
| 输出: 2

::

    def lastRemaining(self, n: int, m: int) -> int:
        i = 0
        array = list(range(n))
        while len(array)>1:
            i = (i + m - 1) % len(array)
            array.pop(i)
        return array[0]

以前很怕这种圆圈的题目....因为不知道循环要怎么做。这道题解法不美妙，纯暴力，纯还原仿真，但是提供了一个很好的思路。

圆圈的题目就用取余 %，判断条件就是 while 
		
找规律&斐波拉契
===================

跳台阶---斐波拉契

剪绳子
----------------------

剑指 Offer 14- I. 

| 给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。
| 请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？
| 例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

| 示例
| 输入: 10
| 输出: 36
| 解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36



数字序列中某一位的数字
-----------------------------
剑指 Offer 44. 

数字以0123456789101112131415…的格式序列化到一个字符序列中。在这个序列中，第5位（从下标0开始计数）是5，第13位是1，第19位是4，等等。

请写一个函数，求任意第n位对应的数字。

::

    def findNthDigit(self, n: int) -> int:
        digit, start, count = 1, 1, 9
        while n > count: # 1.
            n -= count
            start *= 10
            digit += 1
            count = 9 * start * digit
        num = start + (n - 1) // digit # 2.
        return int(str(num)[(n - 1) % digit]) # 3.


.. image:: ../../_static/leetcode/剑指44.png
    :align: center
    :width: 400


https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/solution/mian-shi-ti-44-shu-zi-xu-lie-zhong-mou-yi-wei-de-6/


把数字翻译成字符串
--------------------------
剑指 Offer 46. 

给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。

| 示例
| 输入: 12258
| 输出: 5
| 解释: 12258有5种不同的翻译，分别是"bccfi", "bwfi", "bczi", "mcfi"和"mzi"

::

    def translateNum(self, num: int) -> int:
        num = str(num)
        if len(num)<=1:
            return len(num)
        if len(num)>=2:
            if int(num[:2])<=25:
                res = [1,2]
            else:
                res = [1,1]

        for i in range(2,len(num)):
            if int(num[i-1]+num[i])<=25 and num[i-1]!="0":
                res.append(res[-1] + res[-2])
            else:
                res.append(res[-1])
        return res[-1]
		
num[i-1]!="0" 这里要注意，否则处理 506 这样带0的数据会出错

n个骰子的点数
----------------------
剑指 Offer 60. 

把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。

你需要用一个浮点数数组返回答案，其中第 i 个元素代表这 n 个骰子所能掷出的点数集合中第 i 小的那个的概率。

| 示例 1:
| 输入: 1
| 输出: [0.16667,0.16667,0.16667,0.16667,0.16667,0.16667]

| 示例 2:
| 输入: 2
| 输出: [0.02778,0.05556,0.08333,0.11111,0.13889,0.16667,0.13889,0.11111,0.08333,0.05556,0.02778]

::

    def twoSum(self, n: int) -> List[float]:
        if n==0:
            return []
        probs =[0]*6 + [1]*6
        count = 1
        while count < n:
            temp = []
            for i in range(len(probs)+6):
                left = max(0,i-6)
                right = min(len(probs),i)
                cur = sum(probs[left:right])
                temp.append(cur)
            probs = temp
            count +=1
        res = []
        for x in probs:
            if x > 0:
                res.append(x/(6**n))
        return res

我的憨憨解法思想如下： 下一轮在更新的时候，达到这个值的数量是上一轮前六个数的相加。 

| e.g.
| 第i轮，sum=99能够达到的数量 等于第i-1轮时，sum=93..94...95...96...97...98的求和，因为这些数下一轮加上1~6就能达到99

要特别注意边界条件。所以我在第一轮前面加了六个0，前几轮要判断左边界。 然后在这一轮能够新生成的6个数，要判断右边界。

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
		
		
两个链表的第一个公共节点
--------------------------------
剑指 Offer 52. 

**好题！经典！常看！**

输入两个链表，找出它们的第一个公共节点。

.. image:: ../../_static/leetcode/剑指52.png
    :align: center
    :width: 400

优秀解法::
    
	def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        alist, blist = headA, headB
        while headA != headB:
            if headA:
                headA = headA.next
            else:
                headA = blist
            if headB:
                headB = headB.next
            else:
                headB = alist
        return headA

| 我知道是双指针，然后把两个链表前后拼接在一起，以消除长度不一致的影响。但是，
| 我最开始写的憨憨解法

::

    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        alist, blist = headA, headB
        if not headA or not headB:
            return None
        while headA != headB:
            if headA.next==None and headB.next==None:
                return None
            if headA.next:
                headA = headA.next
            else:
                headA = blist
            if headB.next:
                headB = headB.next
            else:
                headB = alist
        return headA
		
| 有几个坑的地方：
| 1. if headA.next==None and headB.next==None:  return None 这里很重要。
| 不然两个链表完全没有重合结点的时候就会无限循环下去
| 2. 所以我加了 if not headA or not headB:  return None
| 但我最开始写的 return 0. 会在leetcode上面报一个很奇怪的错，int object has not attribute val
| 3. if not headA or not headB:  求求你别再写成 if not headA or headB: 了

所以上面那种优秀的解法完美的避开了我下面的这些坑。如果两个链表完全没有相同的结点，他会循环到两个人都是None，然后headA==headB，返回headA，恰好是None

当然，像他们那样写成这种形式的也可以
::

	def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        node1, node2 = headA, headB
        while node1 != node2:
            node1 = node1.next if node1 else headB
            node2 = node2.next if node2 else headA
        return node1

看着简洁，但是可读性没有最开始的好。我还是建议分开写


位运算
==============
我菜狗，暂时不会

数组中数字出现的次数 II
-----------------------------------
剑指 Offer 56 - II. 

在一个数组 nums 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。

| 示例 1：
| 输入：nums = [3,4,3,3]
| 输出：4

| 示例 2：
| 输入：nums = [9,1,7,9,7,9,7]
| 输出：1


数组中数字出现的次数
-----------------------------
剑指 Offer 56 - I. 

一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。

| 示例 1：
| 输入：nums = [4,1,4,6]
| 输出：[1,6] 或 [6,1]

| 示例 2：
| 输入：nums = [1,2,10,4,1,4,3,3]
| 输出：[2,10] 或 [10,2]