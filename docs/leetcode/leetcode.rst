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


::

    def binary(stand, left, right, potions):
        mid = (left + right) // 2
        if left >= right:
            return left
        if potions[mid] >= stand:
            return binary(stand, left, mid, potions)
        else:
            return binary(stand, mid + 1, right, potions)


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


x的平方根
------------------------------
| leetcode 69. 
| 实现 int sqrt(int x) 函数。
| 计算并返回 x 的平方根，其中 x 是非负整数。
| 由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。

::

    def mySqrt(self, x: int) -> int:
        if x==0 or x==1:
            return x
        l, r = 0, x
        while l <= r:
            mid = (l + r)//2
            if mid**2 == x:
                return mid
            elif mid**2 > x:
                r = mid - 1
            else:
                l = mid + 1
        return r


寻找峰值
--------------------------
leetcode 162. 

峰值元素是指其值严格大于左右相邻值的元素。

给你一个整数数组 nums，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回 任何一个峰值 所在位置即可。

你可以假设 nums[-1] = nums[n] = -∞ 。

你必须实现时间复杂度为 O(log n) 的算法来解决此问题。

| 示例
| 输入：nums = [1,2,1,3,5,6,4]
| 输出：1 或 5 
| 解释：你的函数可以返回索引 1，其峰值元素为 2；或者返回索引 5， 其峰值元素为 6。
::

    def findPeakElement(self, nums: List[int]) -> int:
        def binsearch(left, right):
            if left >= right:
                return left
            mid = (left + right) // 2
            if nums[mid] < nums[mid + 1]:
                return binsearch(mid + 1, right)
            else:
                return binsearch(left, mid)
        return binsearch(0, len(nums) - 1)


O(log n) 暗示了用二分法。但是为什么可以二分呢？上述做法正确的前提有两个：

| 对于任意数组而言，一定存在峰值（一定有解）；
| 二分不会错过峰值。

详细解析看：https://leetcode.cn/problems/find-peak-element/solutions/998441/gong-shui-san-xie-noxiang-xin-ke-xue-xi-qva7v/

排序
====================

.. image:: ../../_static/leetcode/sort.png
    :align: center
    :width: 700


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


另一种解答

https://leetcode.cn/problems/sort-an-array/solution/duo-chong-pai-xu-yi-wang-da-jin-kuai-pai-wgz4/



数组中的逆序对
----------------------
剑指 Offer 51. 

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。

| 输入: [7,5,6,4]
| 输出: 5

::

    def mergeSort(self, nums, tmp, left, right):
        if left >= right:
            return 0
        mid = (left + right) // 2
        inv_count = self.mergeSort(nums, tmp, left, mid) + self.mergeSort(nums, tmp, mid + 1, right)
        i, j, pos = left, mid + 1, left
        while i <= mid and j <= right:
            if nums[i] <= nums[j]:
                tmp[pos] = nums[i]
                i += 1
            else:
                tmp[pos] = nums[j]
                j += 1
                inv_count += mid - i + 1
            pos += 1
        for k in range(i, mid + 1):
            tmp[pos] = nums[k]
            pos += 1
        for k in range(j, right + 1):
            tmp[pos] = nums[k]
            inv_count += mid - i + 1
            pos += 1
        nums[left:right+1] = tmp[left:right+1]
        return inv_count

    def reversePairs(self, nums: List[int]) -> int:
        n = len(nums)
        tmp = [0] * n
        return self.mergeSort(nums, tmp, 0, n - 1)

思路是归并排序。  解法和题解可以看 https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/solution/shu-zu-zhong-de-ni-xu-dui-by-leetcode-solution/  视频讲解不错。

我这个代码和他的略有一点区别。（他的思路是一种解法，代码是另一种解法）。

这个代码和他的思路都是向前看的思想。  他的代码是向后看的思想



需要维护一个队列/单独栈
==================================

柱状图中最大的矩形
-----------------------------
leetcode 84. 

给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

| 示例:
| 输入: [2,1,5,6,2,3]
| 输出: 10

.. image:: ../../_static/leetcode/84.png
    :align: center
    :width: 400

::

    def largestRectangleArea(self, heights: List[int]) -> int:
        ans = heights[0]
        queue = []
        heights = [0] + heights + [0]
        for i in range(len(heights)):
            while queue and heights[i] < heights[queue[-1]]:
                h = heights[queue.pop()]
                w = i - queue[-1] - 1
                ans = max(ans, h * w)
            queue.append(i)
        return ans

.. tip:: 

    这里有几点需要注意的地方：

    1. heights = [0] + heights + [0]  相当于前后加了两个“哨兵”

    2. w = i - queue[-1] - 1  而不是刚刚pop出来的。防止[2, 1, 2]的情况发生，不知道左边界是哪里，因为1会把第一个2给pop掉


.. image:: ../../_static/leetcode/84_2.png
    :align: center
    :width: 700

至于为什么这里是维护一个递增队列，是为了找到以当前这个柱子的高度为最高高度的矩形面积：

.. image:: ../../_static/leetcode/84_3.png



股票价格跨度
----------------------------
leetcode 901. 

设计一个算法收集某些股票的每日报价，并返回该股票当日价格的 跨度 。

当日股票价格的 跨度 被定义为股票价格小于或等于今天价格的最大连续日数（从今天开始往回数，包括今天）。

例如，如果未来 7 天股票的价格是 [100,80,60,70,60,75,85]，那么股票跨度将是 [1,1,1,2,1,4,6] 。

| 实现 StockSpanner 类：
| StockSpanner() 初始化类对象。
| int next(int price) 给出今天的股价 price ，返回该股票当日价格的 跨度 。
 
| 示例：
| 输入：
| ["StockSpanner", "next", "next", "next", "next", "next", "next", "next"]
| [[], [100], [80], [60], [70], [60], [75], [85]]
| 输出：
| [null, 1, 1, 1, 2, 1, 4, 6]

| # Your StockSpanner object will be instantiated and called as such:
| # obj = StockSpanner()
| # param_1 = obj.next(price)
::

    class StockSpanner:
        def __init__(self):
            self.stack = [(0, float(inf))]
            self.day = 0

        def next(self, price: int) -> int:
            self.day += 1
            while price >= self.stack[-1][1]:
                self.stack.pop()
            day = self.stack[-1][0]
            self.stack.append((self.day, price))
            return self.day - day

.. tip:: 

    还是单调栈需要注意的地方！！！其实跟上面一模一样！！！：

    1. self.stack = [(0, float(inf))]  相当于前面加了“哨兵”防止空栈
    
    2. 确定左边的日期标杆时要用栈里的最后一个， 而不是刚刚pop出来的。防止[2, 1, 2, 3]的情况发生，处理3时不知道左边界是哪里


滑动窗口最大值
-----------------------------
| leetcode 239. 

| 给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。

| 返回 滑动窗口中的最大值 。
::

    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        queue = []
        ans = []
        for index, num in enumerate(nums):
            while queue and num >= queue[-1][1]:
                queue.pop()
            if queue and queue[0][0] <= index - k:
                queue.pop(0)
            queue.append((index, num))
            if index - k + 1 >= 0:
                ans.append(queue[0][1])
        return ans
        # 这道题看了解析。https://leetcode.cn/problems/sliding-window-maximum/solution/shuang-xiang-dui-lie-jie-jue-hua-dong-chuang-kou-2/ 维护一个递减队列。里面存index



每日温度
--------------------------
| leetcode 739. 

给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，其中 answer[i] 是指对于第 i 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 0 来代替。
::

    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        length = len(temperatures)
        if length == 1:
            return [0]
        stack = []
        ans = [0] * length
        for index, temp in enumerate(temperatures):
            while stack and temp > stack[-1][1]:
                first = stack.pop()
                ans[first[0]] = index - first[0]
            stack.append((index, temp))
        return ans



下一个更大元素 I
--------------------------
leetcode 496. 

nums1 中数字 x 的 下一个更大元素 是指 x 在 nums2 中对应位置 右侧 的 第一个 比 x 大的元素。

给你两个 没有重复元素 的数组 nums1 和 nums2 ，下标从 0 开始计数，其中nums1 是 nums2 的子集。

对于每个 0 <= i < nums1.length ，找出满足 nums1[i] == nums2[j] 的下标 j ，并且在 nums2 确定 nums2[j] 的 下一个更大元素 。如果不存在下一个更大元素，那么本次查询的答案是 -1 。

返回一个长度为 nums1.length 的数组 ans 作为答案，满足 ans[i] 是如上所述的 下一个更大元素 。
::

    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        store = dict()
        queue = []
        for num in nums2:
            while queue and num > queue[-1]:
                store[queue.pop()] = num
            queue.append(num)
        ans = []
        for num in nums1:
            ans.append(store.get(num, -1))
        return ans


132 模式
-------------------
leetcode 456. 

给你一个整数数组 nums ，数组中共有 n 个整数。132 模式的子序列 由三个整数 nums[i]、nums[j] 和 nums[k] 组成，并同时满足：i < j < k 和 nums[i] < nums[k] < nums[j] 。

如果 nums 中存在 132 模式的子序列 ，返回 true ；否则，返回 false 。

| 示例 1：
| 输入：nums = [1,2,3,4]
| 输出：false
| 解释：序列中不存在 132 模式的子序列。
::

    def find132pattern(self, nums: List[int]) -> bool:
        if len(nums) < 3:
            return False
        k = -float(inf)
        stack = []
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] < k:
                return True
            while stack and stack[-1] < nums[i]:
                k = max(k, stack.pop())
            stack.append(nums[i])
        return False


https://leetcode.cn/problems/132-pattern/solution/xiang-xin-ke-xue-xi-lie-xiang-jie-wei-he-95gt/




滑动窗口
================================

.. Note:: 

   这篇解析写的很好，总结了滑动窗口的全部题目。
   https://leetcode.cn/problems/permutation-in-string/solution/by-flix-ix7f/

   窗口定长，和窗口不定长度是有两种模板的。前面基本是一样的，**把demand字典给统计好**，**有多少个字符串need统计好**

   但是在遍历的时候：

   1. 定长的时候如果big[r]不在demand中，不能直接continue，因为当窗口是此时这样覆盖的时候，big[l]也有可能在demand里面的，是需要对demand[big[l]] 做加减判断的

   不定长的时候，可以continue，因为左边是固定的，还会保留在之前的位置，而不是依赖于右边去做计算


   2. 定长的左边index是确定的，记得l = r - lenp    这里要特别注意这里不需要l = r - lenp + 1，因为是右边左边都要动，此时处理的是左边开始滑动时刻的情况

   不定长的时候，while need <= 0: 再对左边滑出的元素做demand和need的判断


最小覆盖子串
------------------------------
| leetcode 76.

| 给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。

| 注意：
| 对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。
| 如果 s 中存在这样的子串，我们保证它是唯一的答案。
::

    def minWindow(self, s: str, t: str) -> str:
        lens = len(s)
        lent = len(t)
        if lent > lens:
            return ""
        ans = s + "#"
        l = 0
        demand = dict()
        for cha in t:
            demand[cha] = demand.get(cha, 0) + 1
        need = lent
        for r in range(lens):
            if s[r] not in demand:
                continue
            if demand[s[r]] > 0:
                need -= 1
            demand[s[r]] -= 1
            
            while need <= 0:
                if len(ans) > r - l + 1:
                    ans = s[l: r + 1]
                if s[l] in demand:
                    if demand[s[l]] >= 0:
                        need += 1
                    demand[s[l]] += 1
                l += 1
        return ans if len(ans) <= lens else ""


最短超串
------------------------------
| 面试题 17.18.

| 假设你有两个数组，一个长一个短，短的元素均不相同。找到长数组中包含短数组所有的元素的最短子数组，其出现顺序无关紧要。

| 返回最短子数组的左端点和右端点，如有多个满足条件的子数组，返回左端点最小的一个。若不存在，返回空数组。
::

    def shortestSeq(self, big: List[int], small: List[int]) -> List[int]:
        lenb = len(big)
        need = len(small)
        if lenb < need:
            return []
        minlen = lenb + 1
        left = right = minlen
        demand = dict()
        for num in small:
            demand[num] = 1
        l = 0
        for r in range(lenb):
            if big[r] not in demand:
                continue
            if demand[big[r]] == 1:
                need -= 1
            demand[big[r]] -= 1
            while need <= 0:
                if r - l + 1 < minlen:
                    left, right = l, r
                    minlen = r - l + 1
                if big[l] in demand:
                    if demand[big[l]] == 0:
                        need += 1
                    demand[big[l]] += 1
                l += 1
        return [left, right] if minlen <= lenb else []

或者::

    def shortestSeq(self, big: List[int], small: List[int]) -> List[int]:
        i, j = 0, 0
        store = defaultdict(int)
        nums = len(small)
        cnt = 0
        set_small = set(small)
        length = len(big)
        ans = [0, length]
        while j <= length - 1:
            if big[j] in set_small:
                store[big[j]] += 1
                if store[big[j]] == 1:
                    cnt += 1
                while i <= j and cnt == nums:
                    if cnt == nums and j - i < ans[1] - ans[0]:
                        ans = [i, j]
                    if big[i] not in set_small:
                        i += 1
                    else:
                        if store[big[i]] >= 2:
                            store[big[i]] -= 1
                            i += 1
                        else:
                            break
            j += 1
        if ans[1] == length:
            return []
        return ans

    
找到字符串中所有字母异位词
----------------------------------------
| leetcode 438. 
| 给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

| 异位词 指由相同字母重排列形成的字符串（包括相同的字符串）。
::

    def findAnagrams(self, s: str, p: str) -> List[int]:
        lens = len(s)
        lenp = len(p)
        if lens < lenp:
            return []
        ans = []
        demand = defaultdict(int)
        for num in p:
            demand[num] += 1
        need = lenp
        for r in range(lens):
            if s[r] in demand:
                if demand[s[r]] > 0:
                    need -= 1
                demand[s[r]] -= 1
            l = r - lenp # 这里要特别注意这里不需要+1，因为是右边左边都要动，此时处理的是左边开始滑动时刻的情况
            if l >= 0:
                if s[l] in demand:
                    if demand[s[l]] >= 0:
                        need += 1
                    demand[s[l]] += 1
            if need == 0:
                ans.append(r - lenp + 1)
        return ans

.. important:: 
    l = r - lenp    这里要特别注意这里不需要+1，因为是右边左边都要动，此时处理的是左边开始滑动时刻的情况


或者::

    def findAnagrams(self, s: str, p: str) -> List[int]:
        i, j = 0, 0
        demand = defaultdict(int)
        for cha in p:
            demand[cha] += 1
        lens = len(s)
        lenp = len(p)
        needs = len(demand)
        ans = []
        while j <= lens - 1:
            if s[j] in demand:
                demand[s[j]] -= 1
                if demand[s[j]] == 0:
                    needs -= 1
            if j >= lenp:
                if s[i] in demand:
                    demand[s[i]] += 1
                    if demand[s[i]] == 1:
                        needs += 1
                i += 1
            if needs == 0:
                ans.append(i)
            j += 1
        return ans

不要忘记，只要if j >= lenp的时候，i 每次也要+1 ，与是否s[j] in demand没关系



字符串的排列
----------------------------------------
| leetcode 567. 
| 给你两个字符串 s1 和 s2 ，写一个函数来判断 s2 是否包含 s1 的排列。如果是，返回 true ；否则，返回 false 。

| 换句话说，s1 的排列之一是 s2 的 子串 。
::

    def checkInclusion(self, s1: str, s2: str) -> bool:
        lens1 = len(s1)
        lens2 = len(s2)
        if lens2 < lens1:
            return False
        need = lens1
        demand = dict()
        for i in s1:
            demand[i] = demand.get(i, 0) + 1
        for r in range(lens2):
            if s2[r] in demand:
                if demand[s2[r]] > 0:
                    need -= 1
                demand[s2[r]] -= 1

            l = r - lens1
            if l >= 0:
                if s2[l] in demand:
                    if demand[s2[l]] >= 0:
                        need += 1
                    demand[s2[l]] += 1
            if need == 0:
                return True
        return False

这和上一题没区别，是简化版，只需要判断True False。代码不改都能过

209. Minimum Size Subarray Sum
-----------------------------------------
leetcode 209.

Given an array of positive integers nums and a positive integer target, return the minimal length of a subarray whose sum is greater than or equal to target. If there is no such subarray, return 0 instead.

| Example 1:
| Input: target = 7, nums = [2,3,1,2,4,3]
| Output: 2
| Explanation: The subarray [4,3] has the minimal length under the problem constraint.

::

    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        length = len(nums)
        res = length + 1
        i, j = 0, 0
        summ = 0
        while j <= length - 1:
            summ += nums[j]
            while summ >= target:
                res = min(res, j - i + 1)
                if res == 1:
                    return 1
                summ -= nums[i]
                i += 1
            if j <= length - 1 and summ < target:
                j += 1
        return res if res != length + 1 else 0


3. Longest Substring Without Repeating Characters
------------------------------------------------------------------
leetcode 3. 

Given a string s, find the length of the longest substring without repeating characters.

| Example 1:
| Input: s = "abcabcbb"
| Output: 3
| Explanation: The answer is "abc", with the length of 3.

::

    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:
            return 0
        length = len(s)
        store = dict()
        i, j = 0, 0
        store[s[0]] = 0
        ans = 1
        while j <= length - 1:
            if s[j] not in store:
                store[s[j]] = j
                ans = max(ans, j - i + 1)
                j += 1
            else:
                if i == j:
                    j += 1
                    continue
                index = store[s[j]]
                while i <= index:
                    del store[s[i]]
                    i += 1
                store[s[j]] = j
                j += 1
        return ans



30. Substring with Concatenation of All Words
-----------------------------------------------------------------
leetcode 30.

You are given a string s and an array of strings words. All the strings of words are of the same length.

A concatenated substring in s is a substring that contains all the strings of any permutation of words concatenated.

For example, if words = ["ab","cd","ef"], then "abcdef", "abefcd", "cdabef", "cdefab", "efabcd", and "efcdab" are all concatenated strings. "acdbef" is not a concatenated substring because it is not the concatenation of any permutation of words.
Return the starting indices of all the concatenated substrings in s. You can return the answer in any order.

| Example 1:
| Input: s = "barfoothefoobarman", words = ["foo","bar"]
| Output: [0,9]
| Explanation: Since words.length == 2 and words[i].length == 3, the concatenated substring has to be of length 6.
| The substring starting at 0 is "barfoo". It is the concatenation of ["bar","foo"] which is a permutation of words.
| The substring starting at 9 is "foobar". It is the concatenation of ["foo","bar"] which is a permutation of words.
| The output order does not matter. Returning [9,0] is fine too.

简单方法::

    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        store = defaultdict(int)
        all_words_len = len(words) * len(words[0])
        for word in words:
            store[word] += 1
        def check_substrings(substrings):
            temp_store = defaultdict(int)
            for i in range(len(words)):
                word = substrings[i * len(words[0]): (i + 1) * len(words[0])]
                if word in store and temp_store[word] < store[word]:
                    temp_store[word] += 1
                else:
                    return False
            return True
        ans = []
        if len(s) - all_words_len < 0:
            return []
        for i in range(len(s) - all_words_len + 1):
            if check_substrings(s[i:i + all_words_len]):
                ans.append(i)
        return ans

这里有个条件简化，就是所有单词都是一样的长度。这个真是帮大忙了。那么其实就很简单了。先开始统计一下words里面出现的单词及次数，然后在s里面滑动窗口，每个窗口判断是否与words里面出现的单词及次数相同。


优化::

    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        if not words or not s:
            return []

        word_length = len(words[0])
        total_length = word_length * len(words)
        word_count = {}

        # Create a frequency map for words
        for word in words:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

        result = []

        # Check each possible window in the string
        for i in range(word_length):
            left = i
            count = 0
            temp_word_count = {}

            for j in range(i, len(s) - word_length + 1, word_length):
                word = s[j:j + word_length]
                if word in word_count:
                    temp_word_count[word] = temp_word_count.get(word, 0) + 1
                    count += 1

                    while temp_word_count[word] > word_count[word]:
                        left_word = s[left:left + word_length]
                        temp_word_count[left_word] -= 1
                        left += word_length
                        count -= 1

                    if count == len(words):
                        result.append(left)
                else:
                    temp_word_count.clear()
                    count = 0
                    left = j + word_length

        return result
        

为啥就比我写的快这么多呢.......::

    # 我的方法
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        store = defaultdict(int)
        word_len = len(words[0])
        all_words_len = len(words) * word_len
        for word in words:
            store[word] += 1
        def check_substrings(i, j):
            index = i + j * word_len
            temp_store = defaultdict(list)
            for i in range(len(words)):
                word = s[index + i * word_len: index + (i + 1) * word_len]
                if word not in store:
                    return False, i + 1
                elif word in store and len(temp_store[word]) < store[word]:
                    temp_store[word].append(i)
                else:
                    return False, temp_store[word][0] + 1
            return True, 1
        ans = []
        if len(s) - all_words_len < 0:
            return []
        for i in range(word_len):
            times = (len(s) - i) // word_len
            j = 0
            while j <= times:
                flag, steps = check_substrings(i, j)
                if flag:
                    ans.append(i + j * word_len)
                j += steps
        return ans



树的遍历
======================

https://leetcode.cn/problems/binary-tree-preorder-traversal/solution/tu-jie-er-cha-shu-de-si-chong-bian-li-by-z1m/

这个题解里面讲的二叉树说的非常好

.. image:: ../../_static/leetcode/BinaryTree.png
    :align: center
    :width: 700


https://leetcode.cn/problems/same-tree/solution/xie-shu-suan-fa-de-tao-lu-kuang-jia-by-wei-lai-bu-/

这个题解里面提到的比较通用的模板

.. image:: ../../_static/leetcode/BTquestion.png
    :align: center
    :width: 700


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

    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        cur, res, stack = root, [], []
        while cur or stack:
            while cur:
                res.append(cur.val)
                stack.append(cur)
                cur = cur.left
            temp = stack.pop()
            cur = temp.right
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
                res.append(node.val)
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)
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
    
注意！res.append(path[:]) 这里一定要是 path[:]，因为list是可变变量，直接append是浅拷贝，最后res里面只会留下空数组？？？存疑....

和https://leetcode-cn.com/problems/recover-a-tree-from-preorder-traversal/solution/yu-dao-jiu-shen-jiu-xiang-jie-ke-bian-bu-ke-bian-s/说的不太一致


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

199. 二叉树的右视图
---------------------------------
直接层次遍历，取每一层的最后一个就好了......

题解里面很多DFS的....有空再看看


二叉搜索树汇总
--------------------
https://leetcode.cn/problems/same-tree/solution/xie-shu-suan-fa-de-tao-lu-kuang-jia-by-wei-lai-bu-/

.. image:: ../../_static/leetcode/BST.png
    :align: center
    :width: 700



剑指 Offer 54    
二叉搜索树的第k大节点  

剑指 Offer 36    
二叉搜索树与双向链表  

剑指 Offer 33    
二叉搜索树的后序遍历序列  

leetcode 95--99

二叉搜索树的最近公共祖先
----------------------------------------
| leetcode 235. 
| 给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。::

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        a = min(p.val,q.val)
        b = max(p.val,q.val)
        def helper(root,a,b):
            if a<= root.val <= b:
                return root
            elif root.val <a:
                return helper(root.right,a,b)
            else:
                return helper(root.left,a,b)
        if not root:
            return None
        r = helper(root,a,b)
        return r

二叉树的最近公共祖先
------------------------------
| leetcode 236. 
| 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
| 百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
::

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root or p==root or q==root:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if not left and not right:
            return None
        if not left:
            return right
        if not right:
            return left
        return root

这个题解写的很好 https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/solution/236-er-cha-shu-de-zui-jin-gong-gong-zu-xian-hou-xu/  里面的动图解释的很清楚

由于需要先知道左右子树的情况，然后决定向上返回什么。因此「后序遍历」的思想是很关键。

.. image:: ../../_static/leetcode/236.png
    :align: center
    

路径总和 III
------------------------
leetcode 437. 

给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。

路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

.. image:: ../../_static/leetcode/437.png
    :align: center
    :width: 500

本来还觉得我的解法挺好的::

    def helper(node, sumlist):
        if not node:
            return 0
        sumlist = [i + node.val for i in sumlist] + [node.val]
        count = sumlist.count(targetSum)
        return count + helper(node.left, sumlist) + helper(node.right, sumlist)
    return helper(root, [])


后来看了这个，前缀和  https://leetcode.cn/problems/path-sum-iii/solutions/596361/dui-qian-zhui-he-jie-fa-de-yi-dian-jie-s-dey6/
::

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        def dfs(node, presum):
            nonlocal store
            if not node:
                return 0
            presum += node.val
            cnt = store[presum - targetSum]
            store[presum] += 1
            cnt_all = cnt + dfs(node.left, presum) + dfs(node.right, presum)
            store[presum] -= 1
            return cnt_all
        store = defaultdict(int)
        store[0] = 1
        return dfs(root, 0)

所以，其实不用每次遇到一个新的节点，都把所有能得到的组合都列出来。

.. image:: ../../_static/leetcode/437_2.png
    :align: center
    :width: 400

其次，可以用一个字典，记录的是本路径上前缀和出现的次数（关于前缀和可以看leetcode第560题）

然后当完成这个节点的计算时，需要恢复原本状态，就是这个前缀和出现次数-1就行

一开始初始化字典的时候需要 store[0] = 1 因为如果没有这个的话，如果某条路径下全部的前缀和刚好是target，则无法被识别

然后 第八行的 store[presum] += 1 不能放在 cnt = store[presum - targetSum] 前面。  暂时还没想清楚。这个案例过不了  root=[1], tar=0

????？？？？

动态规划
===================



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


连续子数组的最大和/最大子序和
----------------------------------------------
剑指 Offer 42. leetcode 53. （题目一样的）

输入一个整型数组，数组里有正数也有负数。数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

要求时间复杂度为O(n)。

示例1:

| 输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
| 输出: 6
| 解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。

::

    def maxSubArray(self, nums: List[int]) -> int:
        if not nums:
            return -999e999
        ans = temp = -99e99
        for i in range(len(nums)):
            temp = max(temp+nums[i], nums[i])
            ans = max(ans, temp)
        return ans

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




买卖股票的最佳时机
------------------------------
leetcode 121. / 剑指 Offer 63. 

给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。

你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 
::

    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) <= 1:
            return 0
        ans = 0
        temp = float(inf)
        for num in prices:
            temp = min(temp, num)
            ans = max(ans, num - temp)
        return ans
        

买卖股票的最佳时机 II
------------------------------
leetcode 122. 

给你一个整数数组 prices ，其中 prices[i] 表示某支股票第 i 天的价格。

在每一天，你可以决定是否购买和/或出售股票。你在任何时候 最多 只能持有 一股 股票。你也可以先购买，然后在 同一天 出售。

返回 你能获得的 最大 利润 。
::

    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) <= 1:
            return 0
        dp = [[0, 0] for _ in range(len(prices))]
        dp[0][1] = -prices[0]
        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
        return dp[-1][0]


这一题和上一题的区别在于，可以多次买卖。所以不是一锤子交易了

在动态规划的时候，每一天都存在两种情况———————手里有一股，手里清仓了。而当天具体能获得的利润其实取决于昨天的两种状态

因此是一个二维的动态规划。dp[i][0]表示为，第i天手里没有股票了的最大利润；dp[i][1]表示为，第i天手里还有1股的最大利润

| dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])中括号里的解读为：
| 前一天就清仓了 和  昨天还留了一手，今天清仓

| dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
| 前一天还持有1股，今天继续持有 和 昨天清仓了，今天买入1股

这里可以理解为，每次就买卖1股，单价是prices[i]



.. Note::

    注意这里dp需要是dp = [[0, 0] for _ in range(len(prices))] 而不是 dp = [[0, 0] * (len(prices))] 
    
    这样会变成1维数组


这里其实还可以简化：
由于第i天的dp之和第i-1天有关系。可以变成一维数组

::

    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) <= 1:
            return 0
        dp = [0, -prices[0]]
        for i in range(1, len(prices)):
            dp[0], dp[1]= max(dp[0], dp[1] + prices[i]), max(dp[1], dp[0] - prices[i])
        return dp[0]

所以leetcode 714题还要收手续费的话，变化也就是::

    dp[0], dp[1]= max(dp[0], dp[1] + prices[i]), max(dp[1], dp[0] - prices[i] - fee)


买卖股票的最佳时机 III
--------------------------------
leetcode 123. 

给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
::

    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) <= 1:
            return 0
        dp = [[0, -float(inf), -float(inf), -float(inf)] for _ in range(len(prices))]
        dp[0][0] = -prices[0]
        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i - 1][0], -prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i])
            dp[i][2] = max(dp[i - 1][2], dp[i - 1][1] - prices[i])
            dp[i][3] = max(dp[i - 1][3], dp[i - 1][2] + prices[i])
        return max(dp[-1][1], dp[-1][3], 0)

这里dp[i][0、1、2、3]分别指的是 在第i天第一次买、第一次卖、第二次买、第二次卖 时的最大利润


这里其实还有一种解答，暂时还没理解啥意思？？？???
::

    def maxProfit(self, prices: List[int]) -> int:
        ret = [0 for i in range(len(prices))]
        for i in range(2):
            currMaxProfit = 0
            for j in range(1, len(prices)):
                currMaxProfit = max(ret[j], currMaxProfit + prices[j] - prices[j - 1])
                ret[j] = max(ret[j - 1], currMaxProfit)
        return ret[-1]


买卖股票的最佳时机含冷冻期
---------------------------------
leetcode 309.

给定一个整数数组prices，其中第  prices[i] 表示第 i 天的股票价格 。​

设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:

卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。


| 示例
| 输入: prices = [1,2,3,0,2]
| 输出: 3 
| 解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]

::

    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) == 1:
            return 0
        dp = [[0, 0, 0] for _ in range(len(prices))]
        dp[0][2] = -prices[0]
        if len(prices) == 2:
            return max(0, prices[1] - prices[0])
        # 今天没卖但是也没持有, 今天刚卖完， 持有股票
        dp[1][1] = prices[1] - prices[0]
        dp[1][2] = max(-prices[0], -prices[1])
        for i in range(2, len(prices)):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1])
            dp[i][1] = dp[i - 1][2] + prices[i]
            dp[i][2] = max(dp[i - 1][2], dp[i - 1][0] - prices[i])
        return max(dp[-1])

由于当天未持有的状态需要拆分成刚刚卖完和本来就没有。

所以这里每天需要三个空格，分别表示 今天没卖但是也没持有, 今天刚卖完， 持有股票



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

请看下一题：

不同路径 II
---------------------
leetcode 63. 

.. image:: ../../_static/leetcode/63.png
    :align: center
    :width: 400

::

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        if obstacleGrid[0][0]==1:
            return 0
        res = [[0 for _ in range(n)] for _ in range(m)]
        res[0][0]=1
        for i in range(1,m):
            if obstacleGrid[i-1][0]==0 and res[i-1][0]==1 and obstacleGrid[i][0] == 0:
                res[i][0]=1
        for j in range(1,n):
            if obstacleGrid[0][j-1]==0 and res[0][j-1]==1 and obstacleGrid[0][j] == 0:
                res[0][j]=1
        if m==1 or n==1:
            return res[-1][-1]
        for i in range(1,m):
            for j in range(1,n):
                if obstacleGrid[i][j]==1:
                    res[i][j]=0
                else:
                    res[i][j]= res[i-1][j] + res[i][j-1]
        return res[-1][-1]
        
请再看一题：

最小路径和
----------------------
| leetcode 64. 
| 给定一个包含非负整数的 m x n 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
| 说明：每次只能向下或者向右移动一步。
| 示例:
| 输入:
| [
|   [1,3,1],
|   [1,5,1],
|   [4,2,1]
| ]
| 输出: 7
| 解释: 因为路径 1→3→1→1→1 的总和最小。    

::

    def minPathSum(self, grid: List[List[int]]) -> int:
        if len(grid)==1:
            return sum(grid[0])
        if len(grid[0])==1:
            the_sum = 0
            for x in grid:
                the_sum += x[0] 
            return the_sum
        for i in range(1,len(grid)):
            grid[i][0] += grid[i-1][0]
        for j in range(1,len(grid[0])):
            grid[0][j] += grid[0][j-1]
        for i in range(1,len(grid)):
            for j in range(1,len(grid[0])):
                grid[i][j] += min(grid[i-1][j],grid[i][j-1])
        return grid[-1][-1]
        
        
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
-------------------
leetcode 263. 

丑数 就是只包含质因数 2、3 和 5 的正整数。

给你一个整数 n ，请你判断 n 是否为 丑数 。如果是，返回 true ；否则，返回 false 。
::

    def isUgly(self, n: int) -> bool:
        if n <= 0:
            return False
        while n:
            if n % 5 == 0:
                n /= 5
                continue
            elif n % 3 == 0:
                n /= 3
                continue
            elif n % 2 == 0:
                n /= 2
                continue
            
            if n == 1:
                return True
            return False
 

丑数 II
--------------
leetcode 264. / 剑指 Offer 49. 

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

| 注意，这里用三个if的原因是为了解决这个难题：得到6的时候，不仅是2*3，其实也是3*2。所以这两种可能性都要生效，所以这两个指针都要+1

| 再要注意的地方是，我最开始写的是while index<= n。这样算的是第n+1个丑数


Z 字形变换
-----------------
leetcode 6. 

将一个给定字符串根据给定的行数，以从上往下、从左到右进行 Z 字形排列。

.. image:: ../../_static/leetcode/6.png
    :align: center
    :width: 400
    
::

    def convert(self, s: str, numRows: int) -> str:
        if numRows<2:
            return s
        res = ["" for _ in range(numRows)]
        i = 0
        flag = -1
        for n in range(len(s)):
            res[i] += s[n]
            if i==0 or i==numRows-1:
                flag = -flag
            i += flag
        return "".join(res)
    
多巧妙!常看！


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


整数转罗马数字
--------------------
leetcode 12. 

罗马数字包含以下七种字符： I， V， X， L，C，D 和 M。

.. image:: ../../_static/leetcode/12.png
    :align: center
    :width: 400

::

    def intToRoman(self, num: int) -> str:
        search = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), 
        (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
        res = []
        for value,symbol in search:
            count = num//value
            num = num-count*value
            if count>0:
                res.append(symbol*count)
        return "".join(res)    

贪心算法。

其实还有另一种解法，就是按照千位，百位这种的去做。但是情况会复杂很多

联动的下一题：

罗马数字转整数
--------------------
leetcode 13. 

::

    def romanToInt(self, s: str) -> int:
        Roman2Int = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
        Int = 0
        n = len(s)

        for index in range(n - 1):
            if Roman2Int[s[index]] < Roman2Int[s[index + 1]]:
                Int -= Roman2Int[s[index]]
            else:
                Int += Roman2Int[s[index]]

        return Int + Roman2Int[s[-1]]

也还很巧妙

最长公共前缀
---------------------
leetcode 14. 

编写一个函数来查找字符串数组中的最长公共前缀。
如果不存在公共前缀，返回空字符串 ""。

| 示例 1:
| 输入: ["flower","flow","flight"]
| 输出: "fl"

| 示例 2:
| 输入: ["dog","racecar","car"]
| 输出: ""
| 解释: 输入不存在公共前缀。

::

    def longestCommonPrefix(self, strs: List[str]) -> str:
        length = 0
        if strs==[]:
            return ""
        for i in range(len(strs[0])):
            c = strs[0][i]
            for j in range(len(strs)):
                if i>len(strs[j])-1 or strs[j][i]!=c:
                    return strs[0][:length]
            length += 1
        return strs[0]
        
纵向查找。

如果还要优化，可以用二分查找而不是第一个for循环的时候用遍历。
https://leetcode-cn.com/problems/longest-common-prefix/solution/zui-chang-gong-gong-qian-zhui-by-leetcode-solution/        



有效的括号
----------------
| leetcode 20. 
| 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。

| 有效字符串需满足：
| 左括号必须用相同类型的右括号闭合。
| 左括号必须以正确的顺序闭合。

::

    def isValid(self, s: str) -> bool:
        stack = []
        left = ["(","{","["]
        right = {")":"(","}":"{","]":"["}
        for i in range(len(s)):
            if s[i] in left:
                stack.append(s[i])
            elif s[i] in right:
                if len(stack)==0 or stack[-1] != right[s[i]]:
                    return False
                stack.pop()
        if len(stack)>0:
            return False
        return True      

先入后出，用栈就好了。注意字典的生成方式，和最后要判断一下栈是否为空


括号生成
---------------
| leetcode 22. 
| 数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。

| 示例：
| 输入：n = 3
| 输出：[
|        "((()))",
|        "(()())",
|        "(())()",
|        "()(())",
|        "()()()"
|      ]

::

    def generateParenthesis(self, n: int) -> List[str]:
        if n<=0:
            return []
        res = ["()"]
        if n==1:
            return res
        count = 1
        while count<n:
            temp = []
            for i in range(len(res)):
                for j in range(len(res[i])):
                    cur = res[i][:j]+"()"+res[i][j:]
                    temp.append(cur)
            res = list(set(temp))
            count += 1
        return res

我这个解法是动态规划做的。类似剑指 Offer 38. 字符串的排列。其他题解里面很多DFS BFS没太理解


下一个排列
-----------------
leetcode 31. 

| 实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。
| 如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。
| 必须原地修改，只允许使用额外常数空间。
| 以下是一些例子，输入位于左侧列，其相应输出位于右侧列。
| 1,2,3 → 1,3,2
| 3,2,1 → 1,2,3
| 1,1,5 → 1,5,1
::

    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if len(nums)<=1:
            return nums
        pos1 = -1
        for i in range(0,len(nums)-1):
            if nums[i] < nums[i+1]:
                pos1 = i
        if pos1 == -1:
            nums[:] = nums[::-1]
            return 
        pos2 = -1
        for j in range(pos1,len(nums)):
            if nums[j]>nums[pos1]:
                pos2 = j
        nums[pos1], nums[pos2] = nums[pos2], nums[pos1]
        if pos1+1<=len(nums)-1:
            nums[:] = nums[:pos1+1] + nums[pos1+1:][::-1]

思想来自于  https://leetcode-cn.com/problems/next-permutation/solution/xia-yi-ge-pai-lie-by-powcai/

.. image:: ../../_static/leetcode/31.png
    :align: center


外观数列
-------------------
| leetcode 38. 
| 给定一个正整数 n（1 ≤ n ≤ 30），输出外观数列的第 n 项。
| 注意：整数序列中的每一项将表示为一个字符串。
| 「外观数列」是一个整数序列，从数字 1 开始，序列中的每一项都是对前一项的描述。前五项如下：

| 1.     1
| 2.     11
| 3.     21
| 4.     1211
| 5.     111221

| 第一项是数字 1
| 描述前一项，这个数是 1 即 “一个 1 ”，记作 11
| 描述前一项，这个数是 11 即 “两个 1 ” ，记作 21
| 描述前一项，这个数是 21 即 “一个 2 一个 1 ” ，记作 1211
| 描述前一项，这个数是 1211 即 “一个 1 一个 2 两个 1 ” ，记作 111221

::

    def countAndSay(self, n: int) -> str:
        def count_num(last_level):
            count = 1
            num = last_level[0]
            res = ""
            for i in range(1,len(last_level)):
                if last_level[i]==num:
                    count += 1
                else:
                    res = res + str(count) + num
                    num = last_level[i]
                    count = 1
            res = res + str(count) + num
            return res
        level = ["1"]
        if n<=1:
            return "1"
        for i in range(1,n):
            temp = count_num(level[-1])
            level.append(temp)
        return level[-1]


Pow(x, n)
---------------
leetcode 50. 

实现 pow(x, n) ，即计算 x 的 n 次幂函数。

？？？ 找时间再做



跳跃游戏
----------------
| leetcode 55. 
| 给定一个非负整数数组，你最初位于数组的第一个位置。
| 数组中的每个元素代表你在该位置可以跳跃的最大长度。判断你是否能够到达最后一个位置。

| 示例 1:
| 输入: [2,3,1,1,4]
| 输出: true
| 解释: 我们可以先跳 1 步，从位置 0 到达 位置 1, 然后再从位置 1 跳 3 步到达最后一个位置。

| 示例 2:
| 输入: [3,2,1,0,4]
| 输出: false
| 解释: 无论怎样，你总会到达索引为 3 的位置。但该位置的最大跳跃长度是 0 ， 所以你永远不可能到达最后一个位置。

::

    def canJump(self, nums: List[int]) -> bool:
        temp_max = 0 + nums[0]
        for i in range(1,len(nums)):
            if temp_max<i:
                return False
            temp_max = max(temp_max,i+nums[i])
            if temp_max>=len(nums):
                return True
        return True

其实只需要弄明白一件事。只要在遍历的时候，维护一个最远能达到的距离就好了。

假设遍历到了n这个结点，然后n这里最远能走5步，那么从n---n+5都是可以到达的。为什么不怕n-3的时候能走的更远呢？因为已经遍历过了....

请看下一题：

跳跃游戏 II
----------------------
| leetcode 45. 
| 给定一个非负整数数组，你最初位于数组的第一个位置。数组中的每个元素代表你在该位置可以跳跃的最大长度。你的目标是使用最少的跳跃次数到达数组的最后一个位置。

| 示例:
| 输入: [2,3,1,1,4]
| 输出: 2
| 解释: 跳到最后一个位置的最小跳跃数是 2。从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
::

    def jump(self, nums: List[int]) -> int:
        max_arrive = nums[0]
        last_max = nums[0]
        if len(nums)==1:
            return 0
        if max_arrive >= len(nums)-1:
            return 1
        count = 1
        for i in range(1,len(nums)):
            max_arrive = max(max_arrive,i+nums[i])
            if max_arrive >= len(nums)-1:
                return count + 1
            if i==last_max:
                count += 1
                last_max = max_arrive
        return count

特殊情况的讨论稍微有点无聊。这一题比上题多了一步。记录达到上次最远的最少跳跃次数。

从第k步（最远距离）到第k+1步（最远距离）。属于贪心算法的思想




不同路径
------------------------
leetcode 62. 

.. image:: ../../_static/leetcode/62.png
    :align: center
    :width: 400
    
::

    def uniquePaths(self, m: int, n: int) -> int:
        # 数学法不香吗?总共要做出 m+n-2次选择，在这些选择里面有m-1次（或者n-1次）要做出向下走的选择，直接用C啊！
        # C m+n-2 m-1
        def jiecheng(num):
            res = 1
            if num==0:
                return 1
            while num>0:
                res *= num
                num -= 1
            return res
        return int(jiecheng(m+n-2)/(jiecheng(m-1)*jiecheng(m+n-2-m+1)))
        

简化路径
----------------------
| leetcode 71. 
| 以 Unix 风格给出一个文件的绝对路径，你需要简化它。或者换句话说，将其转换为规范路径。
| 在 Unix 风格的文件系统中，一个点（.）表示当前目录本身；此外，两个点 （..） 表示将目录切换到上一级（指向父目录）；两者都可以是复杂相对路径的组成部分。
| 请注意，返回的规范路径必须始终以斜杠 / 开头，并且两个目录名之间必须只有一个斜杠 /。最后一个目录名（如果存在）不能以 / 结尾。此外，规范路径必须是表示绝对路径的最短字符串。
::

    def simplifyPath(self, path: str) -> str:
        temp = path.split("/")
        res = []
        for sym in temp:
            if sym=="":
                continue
            elif sym==".":
                continue
            elif sym=="..":
                if not res:
                    continue
                else:
                    res.pop()
            else:
                res.append(sym+"/")
        result = "".join(res)
        if result.endswith("/"):
            result = result[:-1]
        return "/"+result
        
很愚蠢的题目，直接按照规则一条条来就好了


颜色分类
----------------------
leetcode 75. 

| 给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
| 此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

| 注意:
| 不能使用代码库中的排序函数来解决这道题。
| 示例:
| 输入: [2,0,2,1,1,0]
| 输出: [0,0,1,1,2,2]

| 进阶：
| 一个直观的解决方案是使用计数排序的两趟扫描算法。
| 首先，迭代计算出0、1 和 2 元素的个数，然后按照0、1、2的排序，重写当前数组。
| 你能想出一个仅使用常数空间的一趟扫描算法吗？
::

    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        cur, p0, p2 = 0, 0, len(nums)-1
        if p2==-1:
            return None
        while cur <= p2:
            if nums[cur]==0:
                nums[cur], nums[p0] = nums[p0] , nums[cur]
                p0 += 1
                cur += 1
            elif nums[cur]==1:
                cur += 1
            else:
                nums[cur], nums[p2] = nums[p2] , nums[cur]
                p2 -= 1
                
这道题简直太巧妙了！伪三指针。cur 什么时候要 += 1是精髓！ 请再想想！以及while cur <= p2:

？？？

删除排序数组中的重复项 II
--------------------------------------
leetcode 80. 

给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素最多出现两次，返回移除后数组的新长度。

不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。                            
::

    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0
        i = 1
        dup = 1
        temp = nums[0]
        # for i in range(1,len(nums)):
        while i <= len(nums)-1:
            if nums[i]==temp:
                if dup==1:
                    dup += 1
                    i += 1
                else:
                    del(nums[i])
            else:
                temp = nums[i]
                dup = 1
                i += 1
        return len(nums)

| 思考点：
|  **因为涉及到了del(nums[i])**
| 1. 用while 而不是 for！ 不然的话，i会超出索引，因为range不变的
| 2. 注意，在del的那一步不需要i+=1了，因为已经删除了当前的数
| 3. 一个用来保存当前处理的值，另一个记录duplicate，很巧妙

编辑距离
------------------
leetcode 72

::

    def minDistance(self, word1: str, word2: str) -> int:
        if not word1:
            return len(word2)
        if not word2:
            return len(word1)
        len1 = len(word1)
        len2 = len(word2)
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        for i in range(len2 + 1):
            dp[0][i] = i
        for i in range(len1 + 1):
            dp[i][0] = i
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j])
        return dp[-1][-1]

.. image:: ../../_static/leetcode/72.png
    :align: center
    :width: 550
 
这里 dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]  和 for i in range(len2 + 1):dp[0][i] = i  这几行要搞清楚到底是 len1还是len2！！！！


对“dp[i-1][j-1] 表示替换操作，dp[i-1][j] 表示删除操作，dp[i][j-1] 表示插入操作。”的补充理解：

| 以 word1 为 "horse"，word2 为 "ros"，且 dp[5][3] 为例，即要将 word1的前 5 个字符转换为 word2的前 3 个字符，也就是将 horse 转换为 ros，因此有：
| (1) dp[i-1][j-1]，即先将 word1 的前 4 个字符 hors 转换为 word2 的前 2 个字符 ro，然后将第五个字符 word1[4]（因为下标基数以 0 开始） 由 e 替换为 s（即替换为 word2 的第三个字符，word2[2]）
| (2) dp[i][j-1]，即先将 word1 的前 5 个字符 horse 转换为 word2 的前 2 个字符 ro，然后在末尾补充一个 s，即插入操作
| (3) dp[i-1][j]，即先将 word1 的前 4 个字符 hors 转换为 word2 的前 3 个字符 ros，然后删除 word1 的第 5 个字符


两个字符串的删除操作
--------------------------------
leetcode 583. 

给定两个单词 word1 和 word2 ，返回使得 word1 和  word2 相同所需的最小步数。

每步 可以删除任意一个字符串中的一个字符。

| 示例 1：
| 输入: word1 = "sea", word2 = "eat"
| 输出: 2
| 解释: 第一步将 "sea" 变为 "ea" ，第二步将 "eat "变为 "ea"

::

    def minDistance(self, word1: str, word2: str) -> int:
        dp = [[0] * (len(word1) + 1) for _ in range(len(word2) + 1)]
        for i in range(len(word2) + 1):
            dp[i][0] = i
        for j in range(len(word1) + 1):
            dp[0][j] = j
        
        for i in range(1, len(word2) + 1):
            for j in range(1, len(word1) + 1):
                if word1[j - 1] == word2[i - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i][j - 1], dp[i - 1][j]) + 1
        return dp[-1][-1]


基本类似编辑距离，只不过没有上面的替换功能而已。

.. important:: 

   dp = [[0] * (len(word1) + 1) for _ in range(len(word2) + 1)] 这个地方，第二个len是需要加上range的！！老忘记

   并且，len(word1) + 1 的 +1 不能忘记。总之就是这句话别写错


下面这道题和编辑距离的解题方法很像。


交错字符串
-----------------------
leetcode 97. 

给定三个字符串 s1、s2、s3，请你帮忙验证 s3 是否是由 s1 和 s2 交错 组成的。

两个字符串 s 和 t 交错 的定义与过程如下，其中每个字符串都会被分割成若干 非空 子字符串：

| · s = s1 + s2 + ... + sn
| · t = t1 + t2 + ... + tm

|n - m| <= 1

| 交错 是 s1 + t1 + s2 + t2 + s3 + t3 + ... 或者 t1 + s1 + t2 + s2 + t3 + s3 + ...
| 注意：a + b 意味着字符串 a 和 b 连接。


.. image:: ../../_static/leetcode/97.png
    :align: center
    :width: 550

::

    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        lens1 = len(s1)
        lens2 = len(s2)
        lens3 = len(s3)
        if lens1 + lens2 != lens3:
            return False
        if lens1 == 0 or lens2 == 0:
            return s1 + s2 == s3
        
        dp = [[False] * (lens1 + 1) for _ in range(lens2 + 1)]
        dp[0][0] = True
        for i in range(1, lens2 + 1):
            dp[i][0] = dp[i - 1][0] and s2[i - 1] == s3[i - 1]
        for j in range(1, lens1 + 1):
            dp[0][j] = dp[0][j - 1] and s1[j - 1] == s3[j - 1]

        for i in range(1, lens2 + 1):
            for j in range(1, lens1 + 1):
                dp[i][j] = (dp[i - 1][j] and s2[i - 1] == s3[i + j - 1]) or (dp[i][j - 1] and s1[j - 1] == s3[i + j - 1])
        return dp[-1][-1]


解法：

.. image:: ../../_static/leetcode/97_2.png
    :align: center
    :width: 600


.. image:: ../../_static/leetcode/97_3.png
    :align: center
    :width: 600

https://leetcode.cn/problems/interleaving-string/solution/dong-tai-gui-hua-zhu-xing-jie-shi-python3-by-zhu-3/


**想法：**

1. 其实不需要管题目中的什么s1连着三四个字母拼下去，然后s2连着两个字母拼下去。其实分解成小问题，就是能不能 s1[i−1]==s3[i−1]

2. 以后涉及这种两个字符串一个个去比较，又需要用到动态规划的题目，就把上面那个表格画出来。横纵坐标分别代表什么要搞清楚



不同的子序列
-------------------------
leetcode 115. 

给你两个字符串 s 和 t ，统计并返回在 s 的 子序列 中 t 出现的个数。

题目数据保证答案符合 32 位带符号整数范围。

| 示例 1：

| 输入：s = "rabbbit", t = "rabbit"
| 输出：3
| 解释：
| 如下所示, 有 3 种可以从 s 中得到 "rabbit" 的方案。
| **ra** b **bbit**
| **rab** b **bit**
| **rabb** b **it**

::

    def numDistinct(self, s: str, t: str) -> int:
        m, n = len(t), len(s)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for j in range(n + 1):
            dp[0][j] = 1
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if t[i - 1] != s[j - 1]:
                    dp[i][j] = dp[i][j - 1]
                else:
                    dp[i][j] = dp[i - 1][j - 1] + dp[i][j - 1]
        return dp[-1][-1]


.. image:: ../../_static/leetcode/115.png
    :align: center
    :width: 550


.. image:: ../../_static/leetcode/115_2.png
    :align: center
    :width: 700

一般这种题目都是有套路的，第一行和第一列都是空字符串，做好初始化

在dp的第一行和第一列，想清楚谁是1谁是0，之后遍历的时候就从第二行和第二列开始了！


首先，在t[i-1] 不等于 s[j-1] 时，这个很好理解，那就是dp[i][j] = dp[i][j - 1]，相当于没用上这个新出来的字符串

在t[i-1] 等于 s[j-1] 时。这个理解稍微复杂一点。如上面截图红框处的这个位置。如果用上这个新加的b来匹配需要的b，那么相当于不用之前的内容，为左上角的1
如果不用这个新加的b，那和两个字符不匹配（上面那个一样），为左边的2.所以最后是左上角叠加左边


最长公共子序列
----------------------
leetcode 1143. 

给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。

一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。 两个字符串的 公共子序列 是这两个字符串所共同拥有的子序列。

| 示例 1：
| 输入：text1 = "abcde", text2 = "ace" 
| 输出：3  
| 解释：最长公共子序列是 "ace" ，它的长度为 3 。
::

    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        len1 = len(text1)
        len2 = len(text2)
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
        return dp[-1][-1]


.. image:: ../../_static/leetcode/1143.png
    :align: center
    :width: 650


通配符匹配
-----------------------
leetcode 44. 

给你一个输入字符串 (s) 和一个字符模式 (p) ，请你实现一个支持 '?' 和 '*' 匹配规则的通配符匹配：

| '?' 可以匹配任何单个字符。
| '*' 可以匹配任意字符序列（包括空字符序列）。

判定匹配成功的充要条件是：字符模式必须能够 完全匹配 输入字符串（而不是部分匹配）。

 
| 示例 1：
| 输入：s = "aa", p = "a"
| 输出：false
| 解释："a" 无法匹配 "aa" 整个字符串。

| 示例 2：
| 输入：s = "aa", p = "*"
| 输出：true
| 解释：'*' 可以匹配任意字符串。

::

    def isMatch(self, s: str, p: str) -> bool:
        lens = len(s)
        lenp = len(p)
        dp = [[False] * (lenp + 1) for _ in range(lens + 1)]
        dp[0][0] = True
        flag = True
        for j in range(1, lenp + 1):
            if p[j - 1] == "*" and flag:
                dp[0][j] = True
            elif p[j - 1] != "*":
                flag = False
        
        for i in range(1, lens + 1):
            for j in range(1, lenp + 1):
                if p[j - 1] == s[i - 1] or p[j - 1] == "?":
                    dp[i][j] = dp[i - 1][j - 1]
                elif p[j - 1] == "*":
                    if dp[i][j - 1] or dp[i - 1][j]:
                        dp[i][j] = True
        return dp[-1][-1]

https://leetcode.cn/problems/wildcard-matching/solution/yi-ge-qi-pan-kan-dong-dong-tai-gui-hua-dpsi-lu-by-/


单词接龙
-----------------------
leetcode 127. 

字典 wordList 中从单词 beginWord 和 endWord 的 转换序列 是一个按下述规格形成的序列 beginWord -> s1 -> s2 -> ... -> sk：

| 每一对相邻的单词只差一个字母。
| 对于 1 <= i <= k 时，每个 si 都在 wordList 中。注意， beginWord 不需要在 wordList 中。
| sk == endWord

给你两个单词 beginWord 和 endWord 和一个字典 wordList ，返回 从 beginWord 到 endWord 的 最短转换序列 中的 单词数目 。如果不存在这样的转换序列，返回 0 。
::

    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordset = set(wordList)
        if endWord not in wordset:
            return 0
        queue = [beginWord]
        if beginWord in wordset:
            wordset.remove(beginWord)
        step = 1
        while queue:
            store = []
            for word in queue:
                for i in range(len(word)):
                    # temp = word
                    for j in range(26):
                        temp = word[:i] + chr(j + ord("a")) + word[i + 1:]
                        if temp == endWord:
                            return step + 1
                        if temp in wordset:
                            wordset.remove(temp)
                            store.append(temp)
            step += 1
            queue = store
        return 0


| 这道题其实有两种思路：
| 1.对每个单词，都搜索一遍其他单词，然后遍历该单词的每个字母，记录这两个单词的diff是不是1
| 2.对于每个单词，遍历这个单词的每个字母，从a到z进行替换，看看在不在set里面。

第二种的复杂度是O（N * len Word）。而且可以用BFS来进行优化

bfs：从begin开始，找到所有能替换一个字母就达到的单词，存在queue里面。一层层的来找。最短路径是需要用bfs而不是dfs的

.. image:: ../../_static/leetcode/127.png
    :align: center
    :width: 450

最长上升子序列
---------------------------
| leetcode 300. 
| 给定一个无序的整数数组，找到其中最长上升子序列的长度。

| 示例:
| 输入: [10,9,2,5,3,7,101,18]
| 输出: 4 
| 解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。
::

    def lengthOfLIS(self, nums: List[int]) -> int:
        temp = [1]*len(nums)
        if len(nums)<=1:
            return len(nums)
        for i in range(1,len(nums)):
            for j in range(i):
                if nums[i]>nums[j]:
                    temp[i] = max(temp[i],temp[j]+1)
        return max(temp)
        
.. image:: ../../_static/leetcode/300.png
    :align: center
    :width: 400

打家劫舍
-------------------
| leetcode 198. 
| 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，
如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
| 给定一个代表每个房屋存放金额的非负整数数组，计算你不触动警报装置的情况下，一夜之内能够偷窃到的最高金额。

| 示例 1：
| 输入：[1,2,3,1]
| 输出：4
| 解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。偷窃到的最高金额 = 1 + 3 = 4 。
::

    def rob(self, nums: List[int]) -> int:
        if len(nums)==0:
            return 0
        if len(nums)<=2:
            return max(nums)
        temp = [0]*len(nums)
        temp[0], temp[1] = nums[0], max(nums[0],nums[1])
        for i in range(2,len(nums)):
            temp[i] = max(temp[i-1],temp[i-2]+nums[i])
        return max(temp)
        
思考： 为什么这样可以避免我之前设想的 7，2，3，9  取最前最后的问题呢？ 因为取到2的时候就已经避免这个问题了，因为2比7小，所以根本就没用取2

请看下一题

打家劫舍 II
-----------------------------
| leetcode 213. 
| 你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都围成一圈，这意味着第一个房屋和最后一个房屋是紧挨着的。
同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
给定一个代表每个房屋存放金额的非负整数数组，计算你在不触动警报装置的情况下，能够偷窃到的最高金额。

| 示例 1:
| 输入: [2,3,2]
| 输出: 3
| 解释: 你不能先偷窃 1 号房屋（金额 = 2），然后偷窃 3 号房屋（金额 = 2）, 因为他们是相邻的。

::

    def rob(self, nums: List[int]) -> int:
        def steal(array):
            temp = [0]*len(array)
            temp[0], temp[1] = array[0], max(array[0],array[1])
            for i in range(2,len(array)):
                temp[i] = max(temp[i-1],temp[i-2]+array[i])
            return max(temp)
        if len(nums)==0:
            return 0
        if len(nums)<=2:
            return max(nums)
        return max(steal(nums[1:]),steal(nums[:-1]))

把环拆成两个数组。不取第一个和不取最后一个

打家劫舍 III
-----------------------------------
| leetcode 337. 
| 在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。
一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。
计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。
::

    def rob(self, root: TreeNode) -> int:
        def helper(root):
            if not root:
                return 0, 0
            left = helper(root.left)
            right = helper(root.right)
            v1 = root.val + left[1] + right[1]
            v2 = max(left) + max(right)
            return v1, v2
        return max(helper(root))

计数质数
----------------
| leetcode 204. 
| 统计所有小于非负整数 n 的质数的数量。
| 示例:
| 输入: 10
| 输出: 4
| 解释: 小于 10 的质数一共有 4 个, 它们是 2, 3, 5, 7 。
::

    def countPrimes(self, n: int) -> int:
        res = [2]
        if n<=2:
            return 0
        for i in range(2,n):
            for j in range(len(res)):
                if i%res[j]==0:
                    break
            else:
                res.append(i)
        return len(res)

会有点超时

这种厄拉多塞筛法可以::

    def countPrimes(self, n: int) -> int:
        # 0, 1 不是质数，第一个质数从2开始
        prime = [0,0] + [1]*(n-2)
        for i in range(2,len(prime)):
            if prime[i] ==1:
                prime[i**2::i] = [0] * len(prime[i**2::i])
        return sum(prime)
        # 如果是看谁是质数，就
        # p = [i for i,v in enumerate(prime) if v == 1]
        # return p

| 解释如下：
| i是质数，i的倍数都不是质数
| 写成 [i * 2 :: i] 会有重复的数字 如 ： 2，4，6，8       3，6，9
| prime[i**2::i]的意思是  从i**2 开始，每隔数量i。   

01背包问题
-------------------------
假设有一组物品，w的list是他们的weight，v的list是他们的value，tar是背包的重量。求能带上的最大价值
::

    w = [2,2,6,5,4]
    v = [6,3,5,4,6]
    tar = 10

    # i 是是否用物品
    # j 是背包的重量
    temp = [[0 for _ in range(tar+1)] for _ in range(len(w)+1)]
    for i in range(1,len(w)+1):
        for j in range(1,tar+1):
            if j < w[i-1]:
                temp[i][j] = temp[i-1][j]
            else:
                temp[i][j] = max(temp[i-1][j-w[i-1]]+v[i-1],temp[i-1][j])
            
| 几个要注意的地方：
| 1. 生成temp列表的时候要多一行并且多一列，为了保证i-1和j-w[i-1]不越界
| 2. 由于多生成了一行一列，所以在用下标控制w,v里面的元素的时候，记得-1才是真实的指针
| 3. 转移条件很简单。
|     if 背包重量比当前要判断的重量都要小，那么这个肯定不能取。temp[i][j] = temp[i-1][j]
|     else：判断哪个大：不取这个，还是背包空间为j-w[i-1]时的价值+当前的价值。注意这里一定是temp[i-1][j-w[i-1]]而不是temp[i][j-w[i-1]]。后者意味着这个物品已经取了。
    
请看下一题

分割等和子集
--------------------------
| leetcode 416. 
| 给定一个只包含正整数的非空数组。是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

| 示例 1:
| 输入: [1, 5, 11, 5]
| 输出: true
| 解释: 数组可以分割成 [1, 5, 5] 和 [11].
::

    def canPartition(self, nums: List[int]) -> bool:
        if sum(nums)%2 != 0:
            return False
        tar = sum(nums)//2
        store = [[False for _ in range(tar+1)] for _ in range(len(nums)+1)]
        for i in range(len(nums)+1):
            store[i][0] = True
        for i in range(1,len(nums)+1):
            for j in range(1,tar+1):
                store[i][j] = store[i-1][j]
                if j >= nums[i-1]:
                    store[i][j] = store[i-1][j] or store[i-1][j-nums[i-1]]
        return store[-1][-1]
        
| dp就是用空间换时间。这样别看是俩个for循环，其实时间复杂度是O(nc), c是一半的求和。所以还是On。
| 跟上一题不同的地方是，最开始第一列应该是True
| 解答里面还有些回溯法，理论上更好，这先讨论01背包的解法

目标和
------------------
leetcode 494. 
给定一个非负整数数组，a1, a2, ..., an, 和一个目标数，S。现在你有两个符号 + 和 -。对于数组中的任意一个整数，你都可以从 + 或 -中选择一个符号添加在前面。

返回可以使最终数组和为目标数 S 的所有添加符号的方法数。
::

    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        P = (sum(nums) + S) // 2
        if (sum(nums) + S) % 2 != 0 or sum(nums) < S:
            return 0
        count0 = 0
        while 0 in nums:
            nums.remove(0)
            count0 += 1
        dp = [[0 for _ in range(P+1)] for _ in range(len(nums)+1)]
        for i in range(len(nums)+1):
            dp[i][0] = 1
        for i in range(1, len(nums) + 1):
            for j in range(1, P + 1):
                dp[i][j] = dp[i - 1][j]
                if dp[i-1][j - nums[i-1]] != 0:
                    dp[i][j] += dp[i-1][j-nums[i-1]]
        return dp[-1][-1]*2**(count0)

这个也是模仿的01背包的解法。P = (sum(nums) + S) // 2 这个思路十分的巧妙。

.. image:: ../../_static/leetcode/494.png
    :align: center
    :width: 400

count0这里是会有一些0的情况，0取不取所以乘二

然后这种初始化的方式不可以： dp = [[[1] + [0] * P] * (len(nums) + 1)]

这样的话，* (len(nums) + 1)]部分会变成浅拷贝的复制，下一行改了之后上一行也会变。虽然 [0] * P 这里没问题

https://leetcode-cn.com/problems/target-sum/solution/python-dfs-xiang-jie-by-jimmy00745/   题解里面这种一维的就能解决了，解法也放上来
::

    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        if sum(nums) < S or (sum(nums) + S) % 2 == 1: return 0
        P = (sum(nums) + S) // 2
        dp = [1] + [0 for _ in range(P)]
        for num in nums:
            for j in range(P,num-1,-1):dp[j] += dp[j - num]
        return dp[P]



零钱兑换
----------------------
| leetcode  322. 
| 给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。
| 示例 1:
| 输入: coins = [1, 2, 5], amount = 11
| 输出: 3 
| 解释: 11 = 5 + 5 + 1
| 示例 2:
| 输入: coins = [2], amount = 3
| 输出: -1
::

    def coinChange(self, coins: List[int], amount: int) -> int:
        temp = [float('inf') for _ in range(amount + 1)]
        temp[0] = 0
        for i in range(1, amount + 1):
            for j in range(len(coins)):
                if coins[j] > i:
                    continue
                else:
                    temp[i] = min(temp[i], temp[i - coins[j]] + 1)
        if temp[-1]==float('inf'):
            return -1
        else:
            return temp[-1]
            
| float('inf')这个写法可以借鉴。初始化那里需要关注temp[0] = 0。想想为什么？？？
| 首先初始化，价值为0应该要是0，不然 temp[i - coins[j]] + 1这如果是0----10，会有问题。
| 然后其他的每个值应该是无穷大，因为是取min操作。

最长有效括号
-------------------------
| leetcode 32. 
| 给定一个只包含 '(' 和 ')' 的字符串，找出最长的包含有效括号的子串的长度。

| 示例 1:
| 输入: "(()"
| 输出: 2
| 解释: 最长有效括号子串为 "()"

| 示例 2:
| 输入: ")()())"
| 输出: 4
| 解释: 最长有效括号子串为 "()()"
::

    def longestValidParentheses(self, s: str) -> int:
        res = 0
        stack = [-1]
        for i,v in enumerate(s):
            if s[i]=="(":
                stack.append(i)
            else:
                if len(stack)>1:
                    stack.pop()
                    res = max(res, i-stack[-1])
                else:
                    stack = [i]
        return res


这个题的思路和最开始想的不太一样。因为()括号一左一右是两个，最开始想的是每次遇到右括号就长度加2.但是会遇到种种问题。

所以这里记录一下上一次的非法右括号。 所以初始化的时候是stack = [-1] 。 stack打空了之后再遇到右括号就当成非法右括号，stack = [i]

所以从 i-j+1 变成了 i- j的上一个

单词拆分
----------------------
| leetocde 139. 
| 给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。

| 说明：
| 拆分时可以重复使用字典中的单词。
| 你可以假设字典中没有重复的单词。

| 示例 1：
| 输入: s = "leetcode", wordDict = ["leet", "code"]
| 输出: true
| 解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。

| 示例 2：
| 输入: s = "applepenapple", wordDict = ["apple", "pen"]
| 输出: true
| 解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
| 注意你可以重复使用字典中的单词。

完全背包动态规划解法::

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        dp = [False] * (len(s))
        wordset = set(wordDict)
        for i in range(len(s)):
            for j in range(i + 1, len(s) + 1):
                if i == 0 and s[i:j] in wordset:
                    dp[j - 1] = True
                elif dp[i - 1] and s[i:j] in wordset:
                    dp[j - 1] = True
        return dp[-1]

dfs递归解法::

    def helper(i):
        nonlocal flag
        if flag or i == len(s):
            flag = True
            return True
        if memory_no[i] == 0:
            return False
        for j in range(i + 1, len(s) + 1):
            if s[i:j] in wordset:
                if helper(j):
                    return True
        memory_no[i] = 0
        return False
    wordset = set(wordDict)
    flag = False
    memory_no = [1] * len(s)
    helper(0)
    return flag

这里本来可以很简单的递归，但是这种情况过不了。

| s = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab
| wordDict = ["a","aa","aaa","aaaa","aaaaa","aaaaaa","aaaaaaa","aaaaaaaa","aaaaaaaaa","aaaaaaaaaa"]

所以需要加上记忆化模块，如果在某个index处是到不了的，则需要记下来。


请看下一题：

单词拆分 II （递归中非常重要的一点！强调
-----------------------
| leetcode 140. 
| 给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，在字符串中增加空格来构建一个句子，使得句子中所有的单词都在词典中。返回所有这些可能的句子。

| 说明：
| 分隔时可以重复使用字典中的单词。
| 你可以假设字典中没有重复的单词。

| 示例 1：
| 输入:
| s = "catsanddog"
| wordDict = ["cat", "cats", "and", "sand", "dog"]
| 输出:
| [
|   "cats and dog",
|   "cat sand dog"
| ]

| 示例 2：
| 输入:
| s = "pineapplepenapple"
| wordDict = ["apple", "pen", "applepen", "pine", "pineapple"]
| 输出:
| [
|   "pine apple pen apple",
|   "pineapple pen apple",
|   "pine applepen apple"
| ]

| 解释: 注意你可以重复使用字典中的单词。

::

    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
    
        # 前几行处理特殊情况，但是除了lc上面的测试案例，一般不会这么无聊
        # tmp = set("".join(wordDict))
        # if any([i not in tmp for i in s]):
        #     return []

        dp = [['   ']] + [['']] * len(s)
        store = set(wordDict)
        for i in range(len(s)):
            for j in wordDict:
                if i+1-len(j) >= 0 and j == s[i+1-len(j):i+1] and dp[i+1-len(j)] !=['']:
                    if dp[i + 1] == ['']:
                        dp[i + 1] = [x +' '+ j for x in dp[i + 1 - len(j)]]
                    else:
                        dp[i + 1] += [x +' '+ j for x in dp[i + 1 - len(j)]]
        if dp[-1]==['']:
            return []
        else:
            return [x.strip() for x in dp[-1]]
     
这里的话，跟上一题相比需要保存当前为True的结果。由于输出格式的问题，所以dp里面每个元素用['']保存，第一个多一点空格，最后strip掉就好


递归解法。强调递归中非常重要的一点！！！！！

::

    def helper(path, i):
        # 以i为开头开始计算
        if i >= len(s):
            ans.append(path)
            return True
        if memno[i] == 0:
            return False
        flag = False
        for j in range(i + 1, len(s) + 1):
            if s[i:j] in wordset:
                if helper(path + [s[i:j]], j):
                    flag = True
        if not flag:
            memno[i] = 0
            return False
        return True
    memno = [1] * len(s)
    wordset = set(wordDict)            
    ans = []
    helper([], 0)
    return [" ".join(cont) for cont in ans]


这一题相比于上一题，在递归的时候需要把path也保留下来。一开始我没有写递归函数中最下面的15、16行（return True 和 return False）。这会造成在多重递归的时候，最尾巴的递归由于到了s的终点，能够返回
True，
**但是中间的递归却没有返回True或false给上一层**
，所以上一层没有接收到信号，就变成了默认的None


**递归的时候，如果确定是返回状态，一定要在结尾处也返回状态！！**




为运算表达式设计优先级/ 对表达式添加括号并求值
-----------------------------------------------------------------
| leetcode 241. 
| 给定一个含有数字和运算符的字符串，为表达式添加括号，改变其运算优先级以求出不同的结果。你需要给出所有可能的组合的结果。有效的运算符号包含 +, - 以及 * 。

| 示例 1:
| 输入: "2-1-1"
| 输出: [0, 2]
| 解释: 
| ((2-1)-1) = 0 
| (2-(1-1)) = 2

| 示例 2:
| 输入: "2*3-4*5"
| 输出: [-34, -14, -10, -10, 10]
| 解释: 
| (2*(3-(4*5))) = -34 
| ((2*3)-(4*5)) = -14 
| ((2*(3-4))*5) = -10 
| (2*((3-4)*5)) = -10 
| (((2*3)-4)*5) = 10
::

    def diffWaysToCompute(self, input: str) -> List[int]:
        def helper(arr):
            if arr.isdigit():
                return [int(arr)]
            ans = []
            for i in range(len(arr)):
                if arr[i] in ["+","-","*"]:
                    left = helper(arr[:i])
                    right = helper(arr[i+1:])
                    for x in left:
                        for y in right:
                            if arr[i] == "+":
                                ans.append(x + y)
                            elif arr[i] == "-":
                                ans.append(x - y)
                            else:
                                ans.append(x * y)
            return ans
        return helper(input)


思路：分治思想，分成两部分。遍历到一个符号的时候，可以递归得到左边的全部结果和右边的全部结果。然后看符号是什么，分别相加相减相乘。


和为 K 的子数组
-----------------------
leetcode 560. 

给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的连续子数组的个数 。

子数组是数组中元素的连续非空序列。
::

    def subarraySum(self, nums: List[int], k: int) -> int:
        ans = 0
        store = {0:1}
        temp = 0
        for num in nums:
            temp += num
            if temp - k in store:
                ans += store[temp - k]
            store[temp] = store.get(temp, 0) + 1
        return ans
        # 这道题看了解析的。前缀和+类似2sum的解法真牛啊！


49. Group Anagrams
-------------------------------
leetcode 49.

.. image:: ../../_static/leetcode/49.png
    :width: 450

::

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        store = defaultdict(list)
        for word in strs:
            count = [0] * 26
            for cha in word:
                count[ord(cha) - ord("a")] += 1
            # str_count = ""
            # for i in range(26):
            #     if count[i] != 0:
            #         str_count += chr(i + ord("a")) + str(count[i])
            store[tuple(count)].append(word) # 直接使用tuple
        return list(store.values())    

list不能当字典里的key的时候,使用tuple当字典的key 


128. Longest Consecutive Sequence
-----------------------------------------------
leetcode 128. 

Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.

You must write an algorithm that runs in O(n) time.

| Example 1:
| Input: nums = [100,4,200,1,3,2]
| Output: 4
| Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.

| Example 2:
| Input: nums = [0,3,7,2,5,8,4,6,0,1]
| Output: 9

::

    def longestConsecutive(self, nums: List[int]) -> int:
        max_length = 0
        temp = {}
        for num in nums:
            if num not in temp:
                left = temp.get(num - 1, 0)
                right = temp.get(num + 1, 0)
                length = left + right + 1
                max_length = max(max_length, length)

                temp[num - left] = length
                temp[num + right] = length
                temp[num] = length  # 这里也要更新一下，免得有重复数字
        return max_length

思路是：利用了必须要求连续整数这一特点。当来一个新数时，看看他的左边-1 和右边+1.获得左右的翅膀长度。计算整个的长度，然后要更新左边和右边代表的长度。同时也要更新自己，以免有重复数字，进行重复计算
为什么只需要更新左右端点呢？因为这里判断新数的时候，只会从他的左边-1 和右边+1.获得左右的翅膀长度，不需要从中间找了

这样写是不行的::

    def longestConsecutive(self, nums: List[int]) -> int:
        store = defaultdict(int)
        ans = 0
        for num in nums:
            if num not in store:
                left = store[num - 1]
                right = store[num + 1]
                length = right + left + 1
                store[num - left] = length
                store[num + right] = length
                store[num] = length
                ans = max(ans, length)
        return ans

原因是，这里使用了defaultdict(int)，那么在 left = store[num - 1]的时候，尽管之前left没出现过，但是也会被生成一个0存在字典里面。所以下次要判断left的时候，会被if num not in store拒绝

遍历（Traversal）
=======================
125. Valid Palindrome
-------------------------------------
leetcode 125.

A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.

Given a string s, return true if it is a palindrome, or false otherwise.

| Example 1:
| Input: s = "A man, a plan, a canal: Panama"
| Output: true
| Explanation: "amanaplanacanalpanama" is a palindrome.

::

    def isPalindrome(self, s: str) -> bool:
        i, j = 0, len(s) - 1
        while i <= j:
            if not s[i].isalnum():
                i += 1
            elif not s[j].isalnum():
                j -= 1
            elif s[i].lower() == s[j].lower():
                # 这里要特别注意，为什么这里可以直接用s[i].lower() != s[j].lower() 而不用担心数字
                # 是因为这里s[i] s[j] 永远都是字符串。哪怕s[i]是9，那也是string 9 而不是int 9
                i += 1
                j -= 1                    
            else:              
                return False
        return True


.. important:: 
| 这里要特别注意，为什么这里可以直接用s[i].lower() != s[j].lower() 而不用担心数字。是因为这里s[i] s[j] 永远都是字符串。哪怕s[i]是9，那也是string 9 而不是int 9



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




盛最多水的容器
------------------------
leetcode 11. 

| 给你 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。
| 在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0)。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

.. image:: ../../_static/leetcode/11.png
    :align: center
    :width: 400
    
::

    def maxArea(self, height: List[int]) -> int:
        if len(height)<=1:
            return 0
        l, r = 0, len(height)-1
        res = 0
        while l<r:
            res = max(res,(r-l)*min(height[l],height[r]))
            if height[l]<=height[r]:
                l += 1
            else:
                r -= 1
        return res

典型的双指针


三数之和 &  最接近的三数之和
----------------------------------------
leetcode 15. 和 leetcode 16

给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。

给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，使得它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在唯一答案。

（注意，都是无序的）

都是先排序，再做双指针。第一个for循环是遍历，然后在他后面的元素里面，左指针是左边第一个，右指针是最右边。



区间问题
=======================

合并区间
-------------------
| leetcode 56. 
| 给出一个区间的集合，请合并所有重叠的区间。

| 示例 1:
| 输入: [[1,3],[2,6],[8,10],[15,18]]
| 输出: [[1,6],[8,10],[15,18]]
| 解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].

| 示例 2:
| 输入: [[1,4],[4,5]]
| 输出: [[1,5]]
| 解释: 区间 [1,4] 和 [4,5] 可被视为重叠区间。

::

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if len(intervals)<=1:
            return intervals
        intervals.sort()
        res = [intervals[0]]
        for i in range(1,len(intervals)):
            if intervals[i][0]>res[-1][-1]:
                res.append(intervals[i])
            else:
                res[-1][-1] = max(res[-1][-1],intervals[i][-1])
        return res

或者::

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intvs = sorted(intervals, key = lambda x: x[0])
        ans = []
        start, end = intvs[0][0], intvs[0][1]
        for i in range(1, len(intvs)):
            s, e = intvs[i][0], intvs[i][1]
            if s > end:
                ans.append([start, end])
                start, end = s, e
            else:
                end = max(end, e)
        ans.append([start, end])
        return ans


只要明白一件事就好了，先排序（sort以后先按第一个排序，再按第二个排序）。排序后的列表，如果说新判断的区间，左边的区间都比上一个的右区间大，那么一定不重合

请看下一题：

插入区间
------------------
leetcode 57. 

| 给出一个无重叠的 ，按照区间起始端点排序的区间列表。
| 在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。

| 示例 1:
| 输入: intervals = [[1,3],[6,9]], newInterval = [2,5]
| 输出: [[1,5],[6,9]]

| 示例 2:
| 输入: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
| 输出: [[1,2],[3,10],[12,16]]
| 解释: 这是因为新的区间 [4,8] 与 [3,5],[6,7],[8,10] 重叠。

::

    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        i= 0
        while i<len(intervals) and intervals[i][1]<newInterval[0]:
            i += 1
        if i<=len(intervals)-1:  # 防止i越界
            newInterval[0] = min(newInterval[0],intervals[i][0])
        j = i
        while j<len(intervals) and intervals[j][0]<=newInterval[1]:
            newInterval[1] = max(newInterval[1],intervals[j][1])
            j+=1
        del(intervals[i:j])
        intervals.insert(i,newInterval)
        return intervals
        
请再次深思，为什么i那里是intervals[i][1]<newInterval[0]，而j那里是intervals[j][0]<=newInterval[1]

同时，del的话是可以越界的。比如the_list只有3长度，可以del(the_list[7:9])

::

    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        if not intervals:
            return [newInterval]
        start, end = newInterval[0], newInterval[1]
        if start > intervals[-1][1]:
            return intervals + [newInterval]
        if end < intervals[0][0]:
            return [newInterval] + intervals
        length = len(intervals)
        ans = []
        flag = 0
        for i in range(length):
            s, e = intervals[i][0], intervals[i][1]
            if e < start:
                # not in process
                ans.append([s, e])
            elif s > end:
                # finish
                # record the temp
                if flag == 1:
                    ans.append([start, end])
                    flag = 2
                elif flag == 0:
                    ans.append([start, end])
                    flag = 2
                ans.append([s, e])
            else:
                start = min(s, start)
                end = max(e, end)
                flag = 1
        if flag == 1:
            ans.append([start, end])
        return ans


会议室
-------------------
| leetcode 252
| 给定一个会议时间安排的数组，每个会议时间都会包括开始和结束的时间 [[s1,e1],[s2,e2],...] (si < ei)，请你判断一个人是否能够参加这里面的全部会议。

| 示例 1:
| 输入: [[0,30],[5,10],[15,20]]
| 输出: false
| 示例 2:
| 输入: [[7,10],[2,4]]
| 输出: true
::

    def merge(intervals):
        if len(intervals)==1:
            return True
        intervals.sort()
        for i in range(1,len(intervals)):
            if intervals[i][0]>=intervals[i-1][1]:
                return False
        return True

仿照leetcode56 写的。 没在leetcode上测过....因为这是锁定题目，要收费...

请看下一题

会议室II
---------------------
| leetcode 253 
| 给定一个会议时间安排的数组，每个会议时间都会包括开始和结束的时间 [[s1,e1],[s2,e2],...] (si < ei)，为避免会议冲突，同时要考虑充分利用会议室资源，请你计算至少需要多少间会议室，才能满足这些会议安排。

| 输入：[[0, 30],[5, 10],[15, 20]]
| 输出：2
::

    def minMeetingRooms(self, intervals):
            intervals.sort(key = lambda x: x.start)
            heap = []
            for interval in intervals:
                if heap and interval.start >= heap[0]:
                    heapq.heapreplace(heap, interval.end)                    
                else:
                    heapq.heappush(heap, interval.end)
            return len(heap)
        
这个题也是被锁住了所以就自己写写。原题好像是要用inter.start, inter.end        

跟leetcode 56 一个思路嘛....只不过是 如果重叠就开个新会议室，不重合就能用同一个会议室

无重叠区间
---------------------
leetcode 435. 
给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。

注意:
可以认为区间的终点总是大于它的起点。
区间 [1,2] 和 [2,3] 的边界相互“接触”，但没有相互重叠。
示例 1:
输入: [ [1,2], [2,3], [3,4], [1,3] ]
输出: 1
解释: 移除 [1,3] 后，剩下的区间没有重叠。
::

    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if len(intervals)<=1:
            return 0
        intervals = sorted(intervals, key = lambda x: x[1])
        temp = intervals[0]
        count = 0
        for i in range(1,len(intervals)):
            if intervals[i][0] < temp[1]:
                count += 1
            else:
                temp = intervals[i]
        return count

这里是按照结束时间排序。因为这里的重要性是按照结束时间来定的：选择区间组成无重叠区间，为使区间数量尽可能多，
被选区间的右端点应尽可能小，留给后面的区间的空间就大，那么后面能够选择的区间个数也就大。 贪心思想，其实也有点像是动态规划，到目前为止，我能安排多少个会议。是以截止时间来衡量的。

可以自己拿示例试试。排完序后是[[1, 11], [2, 12], [11, 22], [1, 100]]。那么很明显，第二个[2, 12]是没有用的


如果是以左端点来排序的话，那么情况会复杂一些。需要在有重叠的时候选择右端点最小的那个。
::

    # 这样是错的
    inters = sorted(sorted(intervals, key = lambda x: x[1]), key = lambda x: x[0])

这样肯定不行。这样的意思其实是我最开始的意思，先排左边，左边相同的情况下右边小的为准。例子：[[-100, 100], [-90, -80], [-20, 0], [1, 10]] 其实是需要移除第一个




用最少数量的箭引爆气球
----------------------------------
| leetcode 452. 
| 在二维空间中有许多球形的气球。对于每个气球，提供的输入是水平方向上，气球直径的开始和结束坐标。由于它是水平的，所以y坐标并不重要，因此只要知道开始和结束的x坐标就足够了。
开始坐标总是小于结束坐标。平面内最多存在104个气球。
一支弓箭可以沿着x轴从不同点完全垂直地射出。在坐标x处射出一支箭，若有一个气球的直径的开始和结束坐标为 xstart，xend， 且满足  xstart ≤ x ≤ xend，则该气球会被引爆。
可以射出的弓箭的数量没有限制。 弓箭一旦被射出之后，可以无限地前进。我们想找到使得所有气球全部被引爆，所需的弓箭的最小数量。

| Example:
| 输入:
| [[10,16], [2,8], [1,6], [7,12]]
| 输出:
| 2
| 解释:
| 对于该样例，我们可以在x = 6（射爆[2,8],[1,6]两个气球）和 x = 11（射爆另外两个气球）。
::

    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort(key = lambda x: x[1])
        if len(points) <= 1:
            return len(points)
        count = 1
        end = points[0][1]
        for i in range(1,len(points)):
            if points[i][0] > end:
                count += 1
                end = points[i][1]
        return count

思路和上一题leetcode 435. 很像。就是一定要按照结束时间排序。什么时候开始不重要，什么时候结束才重要。只要后一个的开始小于前一个的结束，则在结束的那一刻一箭都能洞穿

举例如下
::

    ```````````````````````````
    `````````````
        ````````
            `````

这种的是可以一箭洞穿的

汇总区间
-------------------
| leetcode 228. 
| 给定一个无重复元素的有序整数数组，返回数组区间范围的汇总。
| 示例 1:
| 输入: [0,1,2,4,5,7]
| 输出: ["0->2","4->5","7"]
| 解释: 0,1,2 可组成一个连续的区间; 4,5 可组成一个连续的区间。
::

    def summaryRanges(self, nums: List[int]) -> List[str]:
        nums += [float(inf)]
        length = len(nums)
        ans = []
        for i in range(length):
            if i == 0:
                start = nums[i]
                end = nums[i]
            elif nums[i] - nums[i - 1] == 1:
                end = nums[i]
            elif nums[i] - nums[i - 1] > 1:
                if end == start:
                    interval = str(start)
                else:
                    interval = str(start) + "->" + str(end)
                start = nums[i]
                end = nums[i]
                ans.append(interval)
        return ans

leetcode 163 759 986  630



矩阵/二维数组
==================================

旋转二维数组总结
---------------------------------
类似这种题目，一个n x n的二维数组进行旋转。做一个总结。

.. image:: ../../_static/leetcode/rotatematrix.png
    :align: center
    :width: 500

无外乎四种旋转进行组合::

    // 上下对称
    void upDownSymmetry(vector<vector<int>>& matrix) {
        for (int i = 0; i < n/2; ++i) {
            for (int j = 0; j < n; ++j) {
                swap(matrix[i][j], matrix[n-i-1][j]);
            }
        }
    }

    // 左右对称
    void leftRightSymmetry(vector<vector<int>>& matrix) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n/2; ++j) {
                swap(matrix[i][j], matrix[i][n-j-1]);
            }
        }
    }

    // 主对角线对称
    void mainDiagSymmetry(vector<vector<int>>& matrix) {
        for (int i = 0; i < n-1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                swap(matrix[i][j], matrix[j][i]);
            }
        }
    }

    // 副对角线对称
    void subdiagSymmetry(vector<vector<int>>& matrix) {
        for (int i = 0; i < n-1; ++i) {
            for (int j = 0; j < n-i-1; ++j) {
                swap(matrix[i][j], matrix[n-j-1][n-i-1]);
            }
        }
    }

注意这里的 交换，和 i,j的范围

例题解答看下面

旋转图像
----------------
| leetcode 48. 
| 给定一个 n × n 的二维矩阵表示一个图像。
| 将图像顺时针旋转 90 度。
| 说明：
| 你必须在原地旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要使用另一个矩阵来旋转图像。
| 示例 1:
| 给定 matrix = 
| [
|   [1,2,3],
|   [4,5,6],
|   [7,8,9]
| ],

| 原地旋转输入矩阵，使其变为:
| [
|   [7,4,1],
|   [8,5,2],
|   [9,6,3]
| ]

建议使用::

    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        def updown(matrix, n):
            for i in range(n // 2):
                for j in range(n):
                    matrix[i][j], matrix[n - i - 1][j] = matrix[n - i - 1][j], matrix[i][j]
            return matrix

        def diagonal(matrix, n):
            for i in range(n):
                for j in range(i, n):
                    matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
            return matrix
        n = len(matrix)
        matrix = updown(matrix, n)
        matrix = diagonal(matrix, n)
        return matrix

::

    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        for i in range(len(matrix)):
            for j in range(i,len(matrix[0])):
                matrix[i][j],matrix[j][i] = matrix[j][i], matrix[i][j]
        for i in range(len(matrix)):
            matrix[i] = matrix[i][::-1]


这里上下翻转为何不能使用 matrix = matrix[::-1]?
因为这里需要in-place。这种方法会开辟一个额外空间。然后题目还是会去检测之前的matrix所在空间的值



螺旋矩阵/顺时针打印矩阵
------------------------------
leetcode 54. / 剑指 Offer 29. 

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

示例 1：

| 输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
| 输出：[1,2,3,6,9,8,7,4,5]

示例 2：

| 输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
| 输出：[1,2,3,4,8,12,11,10,9,5,6,7]

这种方式更加科学::

    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        up, down, left, right = 0, len(matrix) - 1, 0, len(matrix[0]) - 1
        ans = []
        while True:
            for i in range(left, right + 1):
                ans.append(matrix[up][i])
            up += 1
            if up > down:
                break
            for i in range(up, down + 1):
                ans.append(matrix[i][right])
            right -= 1
            if right < left:
                break
            for i in range(right, left - 1, -1):
                ans.append(matrix[down][i])
            down -= 1
            if up > down:
                break
            for i in range(down, up - 1, -1):
                ans.append(matrix[i][left])
            left += 1
            if left > right:
                break
        return ans


另一种很憨憨的解法，一板一眼的去做::

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


螺旋矩阵 II
------------------------------
| leetcode 59.  
| 给定一个正整数 n，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的正方形矩阵。
| 示例:
| 输入: 3
| 输出:
| [
|  [ 1, 2, 3 ],
|  [ 8, 9, 4 ],
|  [ 7, 6, 5 ]
| ]

::
    
    def generateMatrix(self, n: int) -> List[List[int]]:
        left, right, up, down = 0, n - 1, 0, n - 1
        i, j = 0, 0
        dp = [[0] * n for _ in range(n)]
        num = 1
        while num <= n ** 2:
            for j in range(left, right + 1):
                dp[i][j] = num
                num += 1
            up += 1
            for i in range(up, down + 1):
                dp[i][j] = num
                num += 1
            right -= 1
            for j in range(right, left - 1, -1):
                dp[i][j] = num
                num += 1
            down -= 1
            for i in range(down, up - 1, -1):
                dp[i][j] = num
                num += 1
            left += 1                
        return dp


https://leetcode.cn/problems/spiral-matrix-ii/solution/spiral-matrix-ii-mo-ni-fa-she-ding-bian-jie-qing-x/



289. Game of Life
-----------------------------------------------
According to Wikipedia's article: "The Game of Life, also known simply as Life, is a cellular automaton devised by the British mathematician John Horton Conway in 1970."

The board is made up of an m x n grid of cells, where each cell has an initial state: live (represented by a 1) or dead (represented by a 0). Each cell interacts with its eight neighbors (horizontal, vertical, diagonal) using the following four rules (taken from the above Wikipedia article):

Any live cell with fewer than two live neighbors dies as if caused by under-population.
Any live cell with two or three live neighbors lives on to the next generation.
Any live cell with more than three live neighbors dies, as if by over-population.
Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
The next state is created by applying the above rules simultaneously to every cell in the current state, where births and deaths occur simultaneously. Given the current state of the m x n grid board, return the next state.

.. image:: ../../_static/leetcode/289.png
    :width: 400

::

    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        m = len(board)
        n = len(board[0])
        def check(i, j):
            count = 0
            for (a, b) in [(i + 1, j), (i + 1, j + 1), (i + 1, j - 1), (i, j - 1), (i, j + 1), (i - 1, j), (i - 1, j + 1), (i - 1, j - 1)]:
                if 0 <= a <= m - 1 and 0 <= b <= n - 1 and (board[a][b] == 1 or board[a][b] == 2):
                    count += 1
            return count

        for i in range(m):
            for j in range(n):
                live = check(i, j)
                if board[i][j] == 1 and live < 2:
                    board[i][j] = 2
                if board[i][j] == 1 and live > 3:
                    board[i][j] = 2
                if board[i][j] == 0 and live == 3:
                    board[i][j] = 3
        
        for i in range(m):
            for j in range(n):
                if board[i][j] == 2:
                    board[i][j] = 0
                if board[i][j] == 3:
                    board[i][j] = 1


73. Set Matrix Zeroes
-----------------------------------
Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.

You must do it in place.

.. image:: ../../_static/leetcode/73.png
    :width: 400


如果要用O(1)的空间复杂度::

    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        row = len(matrix)
        col = len(matrix[0])
        row0_flag = 0
        col0_flag = 0
        for j in range(col):
            if matrix[0][j] == 0:
                row0_flag = 1
                break
        for i in range(row):
            if matrix[i][0] == 0:
                col0_flag = 1
                break

        for i in range(1, row):
            for j in range(1, col):
                if matrix[i][j] == 0:
                    matrix[i][0] = matrix[0][j] = 0
        
        for i in range(1, row):
            for j in range(1, col):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        if row0_flag == 1:
            for j in range(col):
                matrix[0][j] = 0
        if col0_flag == 1:
            for i in range(row):
                matrix[i][0] = 0
        
        return matrix


        """
        参考了https://leetcode.cn/problems/set-matrix-zeroes/solution/o1kong-jian-by-powcai/
        为什么要从第二行(1)和第二列(1)开始遍历，这个很重要！！


        因为[0][0]这个位置太重要了，如果只是第一行中间有0，会把第一列也变零
        """



黑白棋翻转
--------------------
| BD笔试题
| 小明最近学会了一种棋，这种棋的玩法和围棋有点类似，最后通过比较黑子和白子所占区域的大小来决定胜负。
| 在下棋过程中，如果白子或者黑子将对方全部围住，则所围区域中的棋子将更换颜色。
| 如果用1表示黑子，0表示白子，给出如下实例：
| 1111
| 0101
| 1101
| 0010
| 因为第2行第3列的白子(0)和第3行第3列的白子(0)完全被黑子(1)围住，因此需要这两个0将变为1.
| 结果变为：
| 1111
| 0111
| 1111
| 0010
| 为了简化问题的求解只需要大家找出所有被黑子围住的白子，并将这些白子变为黑子后输出。
::

    def solution(arr):
        if not arr:
            return []
        queue = []
        m = len(arr)
        for i in range(m):
            if arr[i][0] == 0:
                queue.append((i, 0))
            if arr[i][m - 1] == 0:
                queue.append((i, m - 1))
            if arr[0][i] == 0:
                queue.append((0, i))
            if arr[m - 1][i] == 0:
                queue.append((m - 1, i))

        while queue:
            r, c = queue.pop()
            if 0 <= r < m and 0 <= c < m and arr[r][c] == 0:
                arr[r][c] = "#"
                for (i,j) in [(r + 1, c),(r - 1, c),(r, c + 1),(r, c - 1)]:
                    if 0 <= i < m and 0 <= j < m and arr[i][j]==0:
                        queue.append((i,j))

        for i in range(m):
            for j in range(m):
                if arr[i][j] == 0:
                    arr[i][j] = 1
                elif arr[i][j] == "#":
                    arr[i][j] = 0

        return arr

扫描三次就好了。第一次把四条边上的0计入队列，准备变成'#'

第二次把这些#的四周以及能够蔓延到的地方全部变#

第三次把剩下的0变1，然后把#再变回0.

注意，第二次遍历的时候需要用这个队列，而不能直接从左上到右下去扫描，不然矩阵靠右的会有点问题。 比如有一行是1000，那么第一个0是看不到最右边的#的


好吧其实遍历两次就行。和下题基本一样

被围绕的区域
-----------------------
leetcode 130. 

给你一个 m x n 的矩阵 board ，由若干字符 'X' 和 'O' ，找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。
::

    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        m, n = len(board), len(board[0])
        def inedge(i, j):
            if i == 0 or i == m - 1 or j == 0 or j == n - 1:
                return True
            return False
        def infection(i, j):
            if i < 0 or i > m - 1 or j < 0 or j > n - 1:
                return
            if board[i][j] == "#" or board[i][j] == "X":
                return
            board[i][j] = "#"
            infection(i + 1, j)
            infection(i - 1, j)
            infection(i, j + 1)
            infection(i, j - 1)
        
        for i in range(m):
            for j in range(n):
                if board[i][j] == "O" and inedge(i, j):
                    infection(i, j)
        for i in range(m):
            for j in range(n):        
                if board[i][j] == "O":
                    board[i][j] = "X"
                if board[i][j] == "#":
                    board[i][j] = "O"


从边缘的"O"找起，这些肯定是不能被同化的，标记为"#"。然后开始感染，这些被感染到的也不能被同化，标记为"#"

第二次遍历，判断一下，如果当前是"O"，则被同化为"x"。如果被标记为"#"则还原回"O"。这里注意先后顺序就行


最大矩形
----------------------
| leetcode 85. 
| 给定一个仅包含 0 和 1 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。

| 示例:
| 输入:
| [
|   ["1","0","1","0","0"],
|   ["1","0","1","1","1"],
|   ["1","1","1","1","1"],
|   ["1","0","0","1","0"]
| ]
| 输出: 6
::

    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if not matrix:
            return 0

        def largestRectangleArea(heights):
            heights.append(0)
            stack = [-1]
            max_area = 0
            for i in range(len(heights)):
                while heights[i] < heights[stack[-1]]:
                    h = heights[stack.pop()]
                    w = i - stack[-1] - 1
                    max_area = max(max_area, h*w)
                stack.append(i)
            return max_area

        cur = [0] * len(matrix[0])
        ans = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == "0":
                    cur[j] = 0
                else:
                    cur[j] += 1
            ans = max(ans, largestRectangleArea(cur))
        return ans

这个题需要结合上一题（leetcode 84）

基本思路是这样：以行为单位，遍历到某行的时候，向上看，形成类似上一题的一个个矩形。如果当前是1，那么就一直到上一个0为止，如果本身是0，那么当前位置的矩形也是0。
然后再调用上一题的代码求面积。

https://leetcode-cn.com/problems/maximal-rectangle/solution/zhong-die-fa-kuai-su-jie-ti-by-my10yuan/ 这个老哥的思路不错，摘抄如下：

| 以题例来解释
| [
| ["1","0","1","0","0"],
| ["1","0","1","1","1"],
| ["1","1","1","1","1"],
| ["1","0","0","1","0"]
| ]
| 那么可以按行或者按列去重叠，下面以按行来实现思路:
| 设置一个数组tag，用来记录不同行连续的1的个数

| **首先获得第一行数据：10100**
| 那么记录数组tag = 10100
| --> 此时通过记录数组，可以得到只考虑这一行时，最大的矩阵面积：1*1=1
| --> 因为不同行的连续1最多为1个，不同列的连续1的个数，也是1
| **再获得第二行数据：10111**
| 那么记录数组tag = 20211
| --> 这里的更新方式是，如果row[i]==1,则tag[i]+=1;如果row[i]==0,则tag[i]=0;
| -->因此，只考虑前两行时，最大的矩阵面积：1*3=3
| **再获得第三行数据：11111**
| 那么记录数组tag = 31322
| -->此时只考虑前三行时，最大的矩阵面积：2*3=6
| **再获得第四行数据：10010**
| 那么记录数组tag = 40030
| -->此时只考虑前四行时，最大的矩阵面积：4*1=4
| 所以结果就是 6


最大子矩阵
-----------------------
| 面试题 17.24.    也是某次考试的笔试题，滴滴的面试题
| 给定一个正整数和负整数组成的 N × M 矩阵，编写代码找出元素总和最大的子矩阵。

| 返回一个数组 [r1, c1, r2, c2]，其中 r1, c1 分别代表子矩阵左上角的行号和列号，r2, c2 分别代表右下角的行号和列号。若有多个满足条件的子矩阵，返回任意一个均可。

| 0 -2 -7 0
| 9 2 -6 2
| -4 1 -4 1
| -1 8 0 -2

| 最大子矩阵和为
| 9 2
| -4 1
| -1 8 
::

    def getMaxMatrix(self, matrix: List[List[int]]) -> List[int]:
        def max_1d(array):
            if not array:
                return float('-inf')
            ans = temp = float('-inf')
            start_temp, start_final, end_final = 0, 0, 0
            for i in range(len(array)):
                if temp + array[i] > array[i]:
                    temp = temp + array[i]
                else:
                    start_temp = i
                    temp = array[i]

                if temp > ans:
                    start_final, end_final = start_temp, i
                    ans = temp
            return start_final, end_final, ans

        row = len(matrix)
        col = len(matrix[0])
        maxArea = float('-inf')                     #最大面积
        res = [0, 0, 0, 0]

        for left in range(col):                     #从左到右，从上到下，滚动遍历
            colSum = [0] * row                      #以left为左边界，每行的总和
            for right in range(left, col):          #这一列每一位为右边界
                for i in range(row):                #遍历列中每一位，计算前缀和
                    colSum[i] += matrix[i][right]

                startX, endX, maxAreaCur= max_1d(colSum)#在left，right为边界下的矩阵中，前缀和colSum的最大值
                if maxAreaCur > maxArea:
                    res = [startX, left, endX, right]        #left是起点y轴坐标，right是终点y轴坐标
                    maxArea = maxAreaCur
        return res

.. image:: ../../_static/leetcode/1724.png
    :align: center

为什么是以列来两个指针遍历？ 因为按照行的话不好求和


找规律&斐波拉契&数学
=============================



跳台阶---斐波拉契
-------------------------------


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

请看下一题

解码方法
-----------------------
| leetcode 91. 
| 一条包含字母 A-Z 的消息通过以下方式进行了编码：
| 'A' -> 1
| 'B' -> 2
| ...
| 'Z' -> 26
| 给定一个只包含数字的非空字符串，请计算解码方法的总数。
::

    def numDecodings(self, s: str) -> int:
        if s[0]=="0":
            return 0
        if len(s)==1:
            return 1
        res = [1,2]
        if s[1] == "0" and int(s[0])>2:
            return 0
        if int(s[:2])>26 or s[:2]=="10" or s[:2]=="20":
            res = [1, 1]
        for i in range(2,len(s)):
            if s[i]=="0":
                temp1 = 0
                if s[i-1]=="0" or int(s[i-1])>2:
                    return 0
            else:
                temp1 = res[-1]
            if s[i-1]!="0" and int(s[i-1]+s[i])<=26:
                temp2 = res[-2]
            else:
                temp2 = 0
            res.append(temp1 + temp2)
        return res[-1]
        
情况的讨论还比较复杂，比上一题难在0没有对应，所以初始化包括之后会要多讨论。

而且注意这是字符串，所以s[i]==str(xxx)这里的str不能忘

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



阶乘后的零
-----------------
leetcode 172. 

给定一个整数 n ，返回 n! 结果中尾随零的数量。

提示 n! = n * (n - 1) * (n - 2) * ... * 3 * 2 * 1
::

    def trailingZeroes(self, n: int) -> int:
        ans = 0
        while n >= 1:
            ans += n // 5
            n /= 5
        return int(ans)

参考：https://leetcode.cn/problems/factorial-trailing-zeroes/solution/xiang-xi-tong-su-de-si-lu-fen-xi-by-windliang-3/

注意 这里是末尾的0，所以7!=4020 只有一个0

.. image:: ../../_static/leetcode/172_1.png
    :align: center
    :width: 600

.. image:: ../../_static/leetcode/172_2.png
    :align: center
    :width: 600


链表
===================

前置学习内容
-------------------------
https://www.youtube.com/watch?v=0czlvlqg5xw

这个视频说的很好。链表的题目其实不复杂。基本上就是双指针(快慢指针or前后指针)或者递归. 而且链表只能前进不能后退，是缺点也是优点，简化了很多思路，导致基本上都需要用双指针来解决

有这样一些小技巧：

1. dummy = ListNode(0, head) 虚拟头节点是真的好用

2. 实在不行就需要保存的地方就设置一个变量，然后充分利用python的同时交换的特性，大力出奇迹

反转链表
------------------
leetcode 206./ 剑指 Offer 24.

定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。

示例:

输入: 1->2->3->4->5->NULL

输出: 5->4->3->2->1->NULL


双指针方法一，这个需要当成模板背下来！！！

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
    :width: 300



递归recursion
::

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return
        def helper(cur, pre):
            if not cur:
                return pre
            res = helper(cur.next, cur)
            cur.next = pre
            return res
        return helper(head, None)


这个递归比较难理解。但其实我们应该从后往前看。 假设是12345。这里的res通过层层往下，最后会固定在5这里。我们本身想返回的也就是5这个头。  然后在每一层里面，做的事情就是把后面指向前面.第7行的这个代码其实是阻止我们先反转顺序，而是让我们先找到头。然后再一层层的反转顺序



请看下一题

反转链表 II
--------------------
| leetcode 92. 
| 反转从位置 m 到 n 的链表。请使用一趟扫描完成反转。
| 说明:
| 1 ≤ m ≤ n ≤ 链表长度。
| 示例:
| 输入: 1->2->3->4->5->NULL, m = 2, n = 4
| 输出: 1->4->3->2->5->NULL


方法一：递归(先得看懂上一题)::

    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        def reverseN(node, n):    
            if n == right:
                self.successor = node.next
                return node
            new_head = reverseN(node.next, n + 1)
            node.next.next = node
            node.next = self.successor
            return new_head
        if not head.next:
            return head
        if left == 1:
            return reverseN(head, 1)
        head.next = self.reverseBetween(head.next, left - 1, right - 1)
        return head


参考了 `这个解析 <https://leetcode.cn/problems/reverse-linked-list-ii/solutions/37247/bu-bu-chai-jie-ru-he-di-gui-di-fan-zhuan-lian-biao>`_

如果我们明白了上一个简单题目的递归，那么我们知道，其实递归最后得到的是 "new_head". 

在此基础上，可以将题目稍微升级成 **“反转链表前 N 个节点”**。解决思路差不多，但是需要额外保存一个后续节点successor。因为之前是直接把最后的尾巴指向None了，现在需要指向后续节点successor.

这里 **self.successor** 的写法就很有灵性,把successor当作一个属性，这样整个类中都可以调用。避免到处定义局部变量全局变量

明白这个 **“反转链表前 N 个节点”** 之后，那么问题只需要变成，从哪个节点开始进行这个操作。这个也可以递归解决！本来假设left, right是3，6. 那么挪一挪 以left为1 就会变成1,4

| 

方法二：双指针-头插法::

    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        pre = dummy
        cnt = 0
        while cnt < left - 1:
            pre = pre.next
            cnt += 1
        cur = pre.next
        while cnt < right - 1:
            temp = cur.next
            cur.next, pre.next, temp.next = temp.next, temp, pre.next
            cnt += 1
        return dummy.next

参考了 `这个解析 <https://leetcode.cn/problems/reverse-linked-list-ii/solutions/138910/java-shuang-zhi-zhen-tou-cha-fa-by-mu-yi-cheng-zho>`_

.. image:: ../../_static/leetcode/92.png
    :width: 600

.. image:: ../../_static/leetcode/92_2.png
    :width: 600


主要是对着这个图来进行操作。要先画图出来  

dummy = ListNode(0, head) 虚拟头节点是真的好用, 凡是需要考虑左右边界的问题, 加个虚拟头节点准没错.

实在不行就需要保存的地方就设置一个变量，然后充分利用python的同时交换的特性，大力出奇迹::

    while cnt < right - 1:
        temp1 = cur.next
        temp2 = pre.next
        cur.next, pre.next, temp1.next = temp1.next, temp1, temp2
        cnt += 1

| 
| 


这个是自己很早之前写的代码
::

    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        res = pre = ListNode(-1)
        pre.next = head
        count = 0
        while count<m-1:
            pre = pre.next
            count += 1
        temp1 = pre
        tail = pre.next
        after = None
        pre = pre.next
        count += 1
        while count <= n:
            temp2 = pre
            temp = pre.next
            pre.next = after
            after = pre
            pre = temp
            count += 1
        temp1.next = temp2
        tail.next = temp
        return res.next

| 变量名没取好.....用了很多中间变量保存临时参数
| 题目还是有点难度的，做了挺久。从上一题反转链表引申而来。值得再看看    


141. Linked List Cycle
-----------------------------------
Given head, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.

Return true if there is a cycle in the linked list. Otherwise, return false.

::

    def hasCycle(self, head: Optional[ListNode]) -> bool:
        p = ListNode(0, head)
        q = head
        while q and q.next:
            if p != q:
                p = p.next
                q = q.next.next
            else:
                return True
        return False

简单的快慢指针。注意一开始不要让p q 都是head。不然 [1, 2] 这种情况就直接判定 p == q 然后就true了


两数相加
-------------------
leetcode 2. 

| 给出两个 非空 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 逆序 的方式存储的，并且它们的每个节点只能存储 一位 数字。
| 如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。
| 您可以假设除了数字 0 之外，这两个数都不会以 0 开头。

| 示例：
| 输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
| 输出：7 -> 0 -> 8
| 原因：342 + 465 = 807

::

    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.next = None

    class Solution:
        def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
            res = head = ListNode(0)
            temp = 0
            while l1 or l2 or temp:
                x = l1.val if l1 else 0
                y = l2.val if l2 else 0
                head.next = ListNode((x + y + temp) % 10)
                temp = (x + y + temp) // 10
                if l1:
                    l1 = l1.next
                if l2:
                    l2 = l2.next
                head = head.next
            return res.next



25. Reverse Nodes in k-Group
-----------------------------------------
leetcode 25. 

Given the head of a linked list, reverse the nodes of the list k at a time, and return the modified list.

k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes, in the end, should remain as it is.

You may not alter the values in the list's nodes, only nodes themselves may be changed.

.. image:: ../../_static/leetcode/25.png
    :width: 400

::

    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        def reverse(a, b):  # 翻转从a到b的节点，左闭右开
            pre, cur = None, a
            while cur != b:
                temp = cur.next  
                cur.next = pre
                pre, cur = cur, temp
            return pre

        if not head:
            return None
        a = b = head 
        # b往前k步
        for i in range(k):
            if not b: return head # 不足k个的话说明翻转完成，直接返回head
            b = b.next
        # 翻转从a到b的节点，左闭右开
        new_head = reverse(a, b)
        a.next = self.reverseKGroup(b, k)  # 这里要是a.next而不是b.next 因为已经翻转过，现在尾巴是a
        return new_head


参考思路： https://leetcode.cn/problems/reverse-nodes-in-k-group/solutions/41713/di-gui-si-wei-ru-he-tiao-chu-xi-jie-by-labuladong/

写的真的太好了，要多仔细研究



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

明显的双指针题目。请看下一题


19. Remove Nth Node From End of List
---------------------------------------------------
leetcode 19. 

Given the head of a linked list, remove the nth node from the end of the list and return its head.

Example 1:

| Input: head = [1,2,3,4,5], n = 2
| Output: [1,2,3,5]

::

    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        pre = cur = head
        for i in range(n):
            cur = cur.next
        if not cur:
            return head.next
        while cur.next:
            pre = pre.next
            cur = cur.next
        pre.next = pre.next.next
        return head


合并两个排序的链表
-------------------------
leetcode 21 / 剑指 Offer 25. 

输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。

示例1：

输入：1->2->4, 1->3->4

输出：1->1->2->3->4->4

::
    
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


递归::

    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        def helper(l1, l2):
            if not l1 and not l2:
                return None
            elif not l1:
                return l2
            elif not l2:
                return l1
            elif l1.val <= l2.val:
                node = ListNode(l1.val)
                node.next = helper(l1.next, l2)
            else:
                node = ListNode(l2.val)
                node.next = helper(l1, l2.next)
            return node
        return helper(l1, l2)


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
    
还有下面这道题
    
合并两个有序数组
---------------------------------
leetcode88. 

.. image:: ../../_static/leetcode/88.png
    :align: center
    :width: 400

::

    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        p1, p2 = m-1, n-1
        p = m + n -1
        while p1>=0 and p2>=0:
            if nums1[p1]>=nums2[p2]:
                nums1[p] = nums1[p1]
                p1 -= 1
            else:
                nums1[p] = nums2[p2]
                p2 -= 1
            p -= 1
        if p2>=0:
            nums1[:p2 + 1] = nums2[:p2 + 1]

从后往前排序。三个指针
    

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

环形链表
----------------------------
leetcode 141. 

给你一个链表的头节点 head ，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递 。仅仅是为了标识链表的实际情况。

如果链表中存在环 ，则返回 true 。 否则，返回 false ::

    def hasCycle(self, head: Optional[ListNode]) -> bool:
        p = ListNode(0, head)
        q = head
        while q and q.next:
            if p != q:
                p = p.next
                q = q.next.next
            else:
                return True
        return False

两两交换链表中的节点
---------------------------------
| leetcode 24. 
| 给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
| 你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

| 示例:
| 给定 1->2->3->4, 你应该返回 2->1->4->3.

::

    def swapPairs(self, head: ListNode) -> ListNode:
        i = 0
        res = m = ListNode(0)
        m.next = head
        while m:
            if m.next and m.next.next and i%2==0:
                temp_a = m.next
                temp_b = m.next.next
                temp_bnext = m.next.next.next
                m.next, m.next.next, m.next.next.next = temp_b, temp_a, temp_bnext
            m = m.next
            i+=1
        return res.next


.. image:: ../../_static/leetcode/24.png
    :align: center
    :width: 300
    
碰到这种结点交换的题目，手画一个，然后分清前后关系。最开始做题的时候如果怕做错，就拿temp变量把他们都保存下来。

旋转链表
----------------
leetcode 61. 

给定一个链表，旋转链表，将链表每个节点向右移动 k 个位置，其中 k 是非负数。

| 示例 1:
| 输入: 1->2->3->4->5->NULL, k = 2
| 输出: 4->5->1->2->3->NULL
| 解释:
| 向右旋转 1 步: 5->1->2->3->4->NULL
| 向右旋转 2 步: 4->5->1->2->3->NULL

| 示例 2:
| 输入: 0->1->2->NULL, k = 4
| 输出: 2->0->1->NULL
| 解释:
| 向右旋转 1 步: 2->0->1->NULL
| 向右旋转 2 步: 1->2->0->NULL
| 向右旋转 3 步: 0->1->2->NULL
| 向右旋转 4 步: 2->0->1->NULL

::

    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head or not head.next or k == 0:
            return head
        p = head
        cnt = 1
        while p.next:
            cnt += 1
            p = p.next
        index = k % cnt

        if index == 0:
            return head
        pre, cur = head, head
        for i in range(index):
            cur = cur.next
        while cur and cur.next:
            cur = cur.next
            pre = pre.next
        new_head = pre.next
        pre.next = None
        cur.next = head
        return new_head


第一次遍历获得深度，取余。然后第二次遍历就是双指针了。记得处理一下特殊情况

方法二：首尾相接::

    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if not head:
            return None
        pre = head
        count = 0
        while head and head.next:
            count += 1
            head = head.next

        k = k % count
        move = count - k
        head.next = pre
        while move:
            head = head.next
            pre = pre.next
            move -= 1
        head.next = None
        return pre


删除排序链表中的重复元素 II
------------------------------------------
leetcode 82. 

给定一个排序链表，删除所有含有重复数字的节点，只保留原始链表中 没有重复出现 的数字。

| 示例 1:
| 输入: 1->2->3->3->4->4->5
| 输出: 1->2->5

| 示例 2:
| 输入: 1->1->1->2->3
| 输出: 2->3
::

    def deleteDuplicates(self, head: ListNode) -> ListNode:
        res = ListNode(-1)
        res.next = head
        l, r = res, head
        while r and r.next:
            if l.next.val != r.next.val:
                l = l.next
                r = r.next
            elif l.next.val == r.next.val:
                while r.next and l.next.val == r.next.val:
                    r = r.next
                l.next = r.next
                r = r.next
        return res.next
        
多漂亮的解法！先来个伪头节点，然后双指针。注意while r.next and l.next.val == r.next.val: 这里的循环判断条件

请看下一题

删除排序链表中的重复元素
----------------------------------------
| leetcode 83. 
| 给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。
| 示例 1:
| 输入: 1->1->2
| 输出: 1->2

| 示例 2:
| 输入: 1->1->2->3->3
| 输出: 1->2->3

::

    def deleteDuplicates(self, head: ListNode) -> ListNode:
        pre = ListNode(-1)
        pre.next = head
        l, r = pre, head
        while r and r.next:
            if l.next.val != r.next.val:
                l = l.next
                r = r.next
            else:
                while r.next and l.next.val == r.next.val:
                    r = r.next
                l.next.next = r.next
                l = l.next
                r = r.next
        return pre.next
        
分隔链表
--------------------
| leetcode 86. 
| 给定一个链表和一个特定值 x，对链表进行分隔，使得所有小于 x 的节点都在大于或等于 x 的节点之前。
| 你应当保留两个分区中每个节点的初始相对位置。

| 示例:
| 输入: head = 1->4->3->2->5->2, x = 3
| 输出: 1->2->2->4->3->5

方法一：搞两个listnode队列，分别填充大的和小的::

    def partition(self, head: ListNode, x: int) -> ListNode:
        res1 = small = ListNode(-1)
        res2 = big = ListNode(-1)
        while head:
            if head.val<x:
                small.next = head
                small = small.next
            else:
                big.next = head
                big = big.next
            head = head.next
        big.next = None
        small.next = res2.next
        return res1.next

big.next = None这个不要忘了，不然没有尾结点

方法二：在原本的队列里面使用双指针::

    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        dummy = ListNode(-1, head)
        pre = dummy  # 在这之前的都小于x，队列的尾巴
        cur = dummy    # 用来判断下一个
        while cur.next:
            if cur.next.val < x:
                if pre == cur:  # 如果一直没遇到比x大的节点 就不需要交换节点 更新指针往后就行了
                    cur = cur.next
                    pre = pre.next
                else:
                    temp = cur.next       # 先暂存判断的节点
                    cur.next = temp.next  # cur跳过下一个
                    temp.next = pre.next  # 把temp插入 pre和下一个的中间
                    pre.next = temp       # 把temp插入 pre和下一个的中间
                    pre = temp            # 更新小队列的尾巴pre。
                                          # 但是不能更新cur，因为cur的下一个已经换人了，还未判断过
            else:  # 只需要更新cur
                cur = cur.next
        return dummy.next


这个相对有技术含量。需要认真学习！！！

大致的意思是，pre,cur这俩双指针。pre是小队列的尾巴，那么在pre以及之前的都要小于x, cur永远是用来判断处理下一个的指针。需要注意的是，如果一直没遇到比x大的节点 就不需要交换节点 更新指针往后就行了，这段很重要，我之前没写导致内部循环
然后什么时候该更新pre和cur都要注意，不是每次两个指针都要更新。至于交换，那就是判断的下一个小的节点需要插入到pre的后面。然后原先的cur跳过这个节点就行。


排序链表
-------------------
| leetcode 148. 
| 在 O(n log n) 时间复杂度和常数级空间复杂度下，对链表进行排序。
| 示例 1:
| 输入: 4->2->1->3
| 输出: 1->2->3->4
| 示例 2:
| 输入: -1->5->3->4->0
| 输出: -1->0->3->4->5
::

    def sortList(self, head: ListNode) -> ListNode:
        # 快速排序
        if not head: return None
        # small equal large 的缩写
        # 都指向相应链表的 head
        s = e = l = None
        target = head.val
        while head:
            nxt = head.next
            if head.val>target:
                head.next = l
                l = head
            elif head.val==target:
                head.next = e
                e = head
            else:
                head.next = s
                s = head
            head = nxt
        
        s = self.sortList(s)
        l = self.sortList(l)
        # 合并 3 个链表
        dummy = ListNode(0)
        cur = dummy # cur: 非 None 的尾节点
        # p: 下一个需要连接的节点
        for p in [s, e, l]:
            while p:
                cur.next = p
                p = p.next
                cur = cur.next
        return dummy.next

        
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


格雷编码
----------------
| leetcode 89. 
| 格雷编码是一个二进制数字系统，在该系统中，两个连续的数值仅有一个位数的差异。
| 给定一个代表编码总位数的非负整数 n，打印其格雷编码序列。即使有多个不同答案，你也只需要返回其中一种。
| 格雷编码序列必须以 0 开头。
::

    def grayCode(self, n: int) -> List[int]:
        res = ["0","1"]
        if n == 0:
            return [0]
        for i in range(1,n):
            temp0 = []
            temp1 = []
            for j in range(len(res)):
                temp0.append("0"+res[j])
                temp1.append("1"+res[::-1][j])
            res = temp0 + temp1
        result = [int(x,2) for x in res]
        return result

我这样做不是位运算，是动态规划
        
解法的思想来自 https://leetcode-cn.com/problems/gray-code/solution/gray-code-jing-xiang-fan-she-fa-by-jyd/

.. image:: ../../_static/leetcode/89.png
    :align: center
    
回溯
=====================
思路模仿自 https://leetcode-cn.com/problems/permutations/solution/hui-su-suan-fa-xiang-jie-by-labuladong-2/   

**回溯模板**

回溯模板
------------------

::

    result = []
    def backtrack(路径, 选择列表):
        if 满足结束条件:
            result.add(路径)
            return
        
        for 选择 in 选择列表:
            做选择
            backtrack(路径, 选择列表)
            撤销选择



https://leetcode-cn.com/problems/combination-sum-ii/solution/hui-su-xi-lie-by-powcai/

leetcode 60. 第k个排列

组合总和
-----------------
| leetcode 39.
| 给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
| candidates 中的数字可以无限制重复被选取。
| 说明：
| 所有数字（包括 target）都是正整数。
| 解集不能包含重复的组合。 
| 示例 1：
| 输入：candidates = [2,3,6,7], target = 7,
| 所求解集为：
| [
|   [7],
|   [2,2,3]
| ]
::

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res, path = [], []
        candidates.sort()
        start = 0
        def calculate(res,path,start):
            for i in range(start,len(candidates)):
                path.append(candidates[i])
                if sum(path)==target:
                    res.append(path[:])
                    path.pop()
                    break
                elif sum(path)>target:
                    path.pop()
                    break
                calculate(res,path,i)
                path.pop()
        calculate(res,path,start)
        return res
        
倒数第三行的path.pop()值得再好好思考！

用标准的回溯模板写出来::

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res, path = [], []
        candidates.sort()
        start = 0
        def calculate(res,path,start):
            if sum(path)==target:
                res.append(path[:])
                return
            for i in range(start,len(candidates)):
                path.append(candidates[i])
                if sum(path)>target:
                    path.pop()
                    break
                calculate(res,path,i)
                path.pop()
        calculate(res,path,start)
        return res


组合总和 II
-------------------------
| leetcode 40. 
| 给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
| candidates 中的每个数字在每个组合中只能使用一次。
| 说明：
| 所有数字（包括目标数）都是正整数。
| 解集不能包含重复的组合。 
| 示例 1:
| 输入: candidates = [10,1,2,7,6,1,5], target = 8,
| 所求解集为:
| [
|   [1, 7],
|   [1, 2, 5],
|   [2, 6],
|   [1, 1, 6]
| ]
::

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res, path = [], []
        candidates.sort()
        def findit(res,path,start):
            for i in range(start,len(candidates)):
                if i>start and candidates[i]==candidates[i-1]:
                    continue
                path.append(candidates[i])
                total = sum(path)
                if total==target:
                    res.append(path[:])
                    path.pop()
                    break
                elif total > target:
                    path.pop()
                    break
                findit(res,path,i+1)
                path.pop()
        findit(res,path,0)
        return res

if i>start and candidates[i]==candidates[i-1]:这里要仔细思考

按照标准的回溯模板来写::

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res, path = [], []
        candidates.sort()
        def findit(res,path,start):
            if sum(path)==target:
                res.append(path[:])
                return
            for i in range(start,len(candidates)):
                if i>start and candidates[i]==candidates[i-1]:
                    continue
                path.append(candidates[i])
                if sum(path) > target:
                    path.pop()
                    break
                findit(res,path,i+1)
                path.pop()
        findit(res,path,0)
        return res
        
        
全排列
---------------------------------
| leetcode 46. 
| 给定一个 没有重复 数字的序列，返回其所有可能的全排列。
| 示例:
| 输入: [1,2,3]
| 输出:
| [
|   [1,2,3],
|   [1,3,2],
|   [2,1,3],
|   [2,3,1],
|   [3,1,2],
|   [3,2,1]
| ]
::

    def permute(self, nums: List[int]) -> List[List[int]]:
        res, path = [], []
        def all_sort(res, path):
            if len(path)==len(nums):
                res.append(path[:])
                return 
            for i in range(0,len(nums)):
                if nums[i] not in path:
                    path.append(nums[i])
                    all_sort(res, path)
                    path.pop()
        all_sort(res, path)
        return res
        
        
全排列 II
-----------------
| leetcode 47. 
| 给定一个可包含重复数字的序列，返回所有不重复的全排列。
| 示例:
| 输入: [1,1,2]
| 输出:
| [
|   [1,1,2],
|   [1,2,1],
|   [2,1,1]
| ]
::

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res, path = [], []
        nums.sort()
        used = [0 for i in range(len(nums))]
        def backtrack(res,path,used):
            if len(path)==len(nums):
                res.append(path[:])
                return
            for i in range(len(nums)):
                if used[i] == 1:
                    continue
                if i>0 and nums[i]==nums[i-1] and used[i-1]:
                    continue
                path.append(nums[i])
                used[i] = 1
                backtrack(res,path,used)
                path.pop()
                used[i] = 0
        backtrack(res,path,used)
        return res
        
关键点         

if used[i] == 1:

if i>0 and nums[i]==nums[i-1] and used[i-1]:        

组合
------------------
| leetcode 77. 
| 给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。
| 示例:
| 输入: n = 4, k = 2
| 输出:
| [
|   [2,4],
|   [3,4],
|   [2,3],
|   [1,2],
|   [1,3],
|   [1,4],
| ]
::

    def combine(self, n: int, k: int) -> List[List[int]]:
        nums = [x for x in range(1,n+1)]
        res, path = [], []
        def backtrack(res, path, start):
            if len(path)==k:
                res.append(path[:])
                return
            for i in range(start, len(nums)):
                path.append(nums[i])
                backtrack(res, path, i+1)
                path.pop()
        backtrack(res, path, 0)
        return res
        
        
子集
----------------
| leetcode 78. 
| 给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
| 说明：解集不能包含重复的子集。
| 示例:
| 输入: nums = [1,2,3]
| 输出:
| [
|   [3],
|   [1],
|   [2],
|   [1,2,3],
|   [1,3],
|   [2,3],
|   [1,2],
|   []
| ]

::

    def subsets(self, nums: List[int]) -> List[List[int]]:
        res, path = [], []
        def backtrack(res, path, length, start):
            if len(path)==length:
                res.append(path[:])
                return
            for i in range(start,len(nums)):
                path.append(nums[i])
                backtrack(res, path, length, i+1)
                path.pop()
        for i in range(0,len(nums)+1):
            backtrack(res, path, i, 0)
        return res
        
思想是借鉴的上一题。不算很优秀的解法，因为重复点在于最后那个for循环，并没有用上上一次循环的结果

为什么官方的那个解答也是这么写的.......以后有时间再优化吧

子集 II
-------------------------
| leetcode 90. 
| 给定一个可能包含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
| 说明：解集不能包含重复的子集。
| 示例:
| 输入: [1,2,2]
| 输出:
| [
|   [2],
|   [1],
|   [1,2,2],
|   [2,2],
|   [1,2],
|   []
| ]
::

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        def backtrack(path, res, start, length):
            if len(path)==length:
                res.append(path[:])
                return
            for i in range(start,len(nums)):
                if i>start and nums[i]==nums[i-1]:
                    continue
                path.append(nums[i])
                backtrack(path, res, i+1, length)
                path.pop()
        for length in range(0,len(nums)+1):
            path = []
            backtrack(path, res, 0, length)
        return res
        
        
单词搜索
----------------------
| leetcode 79. 
| 给定一个二维网格和一个单词，找出该单词是否存在于网格中。
| 单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
| 示例:
| board =
| [
|   ['A','B','C','E'],
|   ['S','F','C','S'],
|   ['A','D','E','E']
| ]
| 给定 word = "ABCCED", 返回 true
| 给定 word = "SEE", 返回 true
| 给定 word = "ABCB", 返回 false
::

    def exist(self, board: List[List[str]], word: str) -> bool:
        directions = [(1,0),(-1,0),(0,1),(0,-1)]
        used = [[0 for j in range(len(board[0]))] for i in range(len(board))]
        def search(i,j,index):
            if index==len(word)-1:
                return board[i][j]==word[index]
            if board[i][j]==word[index]:
                used[i][j] = 1
                for direct in directions:
                    new_i = i + direct[0]
                    new_j = j + direct[1]
                    if 0<=new_i<=len(board)-1 and 0<=new_j<=len(board[0])-1 and not used[new_i][new_j] and search(new_i,new_j,index+1):
                        return True
                used[i][j] = 0
                return False
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                if search(i,j,0):
                    return True
        return False
        
这个还真有点难....回溯法的集成

图
====================
课程表
-----------------
| leetcode 207. 
| 你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。

| 在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。

| 例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
| 请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。
::

    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        first_second = defaultdict(list)
        demand = defaultdict(int)
        qualified = list()
        for key in prerequisites:
            first_second[key[1]].append(key[0])
            demand[key[0]] += 1
        for course in range(numCourses):
            if course not in demand:
                qualified.append(course)
        
        while qualified:
            cur = qualified.pop()
            numCourses -= 1
            if cur in first_second:
                for second in first_second[cur]:
                    demand[second] -= 1
                    if demand[second] == 0:
                        qualified.append(second)
        return numCourses == 0


# 参考了解析https://leetcode.cn/problems/course-schedule/solution/bao-mu-shi-ti-jie-shou-ba-shou-da-tong-tuo-bu-pai-/
# 使用拓扑的入度与出度


课程表 II
------------------------------
leetcode 210. 

现在你总共有 numCourses 门课需要选，记为 0 到 numCourses - 1。给你一个数组 prerequisites ，其中 prerequisites[i] = [ai, bi] ，表示在选修课程 ai 前 必须 先选修 bi 。

例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示：[0,1] 。

返回你为了学完所有课程所安排的学习顺序。可能会有多个正确的顺序，你只要返回 任意一种 就可以了。如果不可能完成所有课程，返回 一个空数组 。
::

    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        ans = []
        queue = []
        demand = defaultdict(int)
        relation = defaultdict(list)
        for course in prerequisites:
            demand[course[0]] += 1
            relation[course[1]] += [course[0]]
        for cla in range(numCourses):
            if cla not in demand:
                queue.append(cla)
        while queue:
            cla = queue.pop()
            ans.append(cla)
            numCourses -= 1
            for after in relation[cla]:
                demand[after] -= 1
                if demand[after] == 0:
                    queue.append(after)
        for cla in demand:
            if demand[cla] > 0:
                return []
        return ans



岛屿数量
-----------------------
| leetcode 200. 
| 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
| 岛屿总是被水包围，并且每座岛屿只能由水平方向或竖直方向上相邻的陆地连接形成。
| 此外，你可以假设该网格的四条边均被水包围。

| 示例 1:
| 输入:
| [
| ['1','1','1','1','0'],
| ['1','1','0','1','0'],
| ['1','1','0','0','0'],
| ['0','0','0','0','0']
| ]
| 输出: 1
::

    def dfs(self,grid,i,j):
        grid[i][j]=0
        row, col = len(grid), len(grid[0])
        next_step = [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]
        for new_x, new_y in next_step:
            if 0<=new_x<=row-1 and 0<=new_y<=col-1 and grid[new_x][new_y]=="1":
                self.dfs(grid,new_x,new_y)

    def numIslands(self, grid: List[List[str]]) -> int:
        ans = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]=="1":
                    ans +=1
                    self.dfs(grid,i,j)
        return ans
        
| 思想就是：遍历，然后DFS。注意DFS的时候不要越界-------0<=new_x<=row-1 and 0<=new_y<=col-1
| 这个写法很不错：
|         next_step = [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]
|         for new_x, new_y in next_step:

python小知识点运用
============================

最大数--sort的key=cmp_to_key写法
---------------------------------------------
leetcode 179. 

给定一组非负整数 nums，重新排列每个数的顺序（每个数不可拆分）使之组成一个最大的整数。

注意：输出结果可能非常大，所以你需要返回一个字符串而不是整数。

| 示例 1：
| 输入：nums = [10,2]
| 输出："210"

| 示例 2：
| 输入：nums = [3,30,34,5,9]
| 输出："9534330"
::

    def largestNumber(self, nums: List[int]) -> str:
        def fun(x, y):
            if x + y > y + x:
                return 1
            return -1
        nums = list(map(str, nums))
        nums.sort(key=cmp_to_key(fun), reverse=True)
        return "0" if nums[0] == "0" else "".join(nums)


这里考察的是sort的key=cmp_to_key写法。这样可以自定义如何来排序，如何比较两个值。

如果只是一个值，比如 nums = [("aa", 1), ("bb", 3), ("cc", 2)]可以

nums.sort(key=lambda x: x[1]) 按照括号第二位进行排列

辗转相除法
------------------------
::

    def gcd(a, b):
        while b != 0:
            a, b = b, a % b
        return a

看下面两题

字符串的最大公因子---辗转相除法
-----------------------------------------------------
leetcode 1071. 

对于字符串 s 和 t，只有在 s = t + ... + t（t 自身连接 1 次或多次）时，我们才认定 “t 能除尽 s”。

给定两个字符串 str1 和 str2 。返回 最长字符串 x，要求满足 x 能除尽 str1 且 x 能除尽 str2 。

| 示例 1：
| 输入：str1 = "ABCABC", str2 = "ABC"
| 输出："ABC"

| 示例 2：
| 输入：str1 = "ABABAB", str2 = "ABAB"
| 输出："AB"
::

    def gcdOfStrings(self, str1: str, str2: str) -> str:
        if str1 + str2 != str2 + str1:
            return ""
        def gcd(a, b):
            while b != 0:
                a, b = b, a % b
            return a
        k = gcd(len(str1), len(str2))
        return str1[:k]


这里判断两个字符串是否有“最大公约数”是通过str1 + str2 == str2 + str1。这个有些意想不到。直观上符合认知
假设str1和str2的最大公约数为x，比如分别由 xxx 和 xxxxx 组成，那str1 + str2 == str2 + str1

这里还使用到了辗转相除法


直线上最多的点数
------------------------
leetcode 149. 

给你一个数组 points ，其中 points[i] = [xi, yi] 表示 X-Y 平面上的一个点。求最多有多少个点在同一条直线上。

.. image:: ../../_static/leetcode/149.png
    :align: center
    :width: 400


https://leetcode.cn/problems/max-points-on-a-line/solution/gong-shui-san-xie-liang-chong-mei-ju-zhi-u44s/

非常规题
======================

概率问题见 machine_learning 概率论
-----------------------------------------------
概率问题见 machine_learning 概率论

用小随机数生成大随机数
------------------------------------------
比如 用一个能生成[1, 5]的随机数生成器X，构建一个能生成[1, 10]的随机数生成器Y

或者用一个6面的筛子给15个人分月饼。

这种的就是编码就好了。

比如用一个6面的筛子给15个人分月饼：

两个数位分别可以实现6*6=36种情况，36/15 = 2 ......6 所以每两种情况编码一位。如果摇到分配不到的空值就重新摇

超多数字，从中找出只出现过一次的数字
--------------------------------------------------
.. image:: ../../_static/leetcode/超多数字.png
    :align: center

海量数据处理面试题
-------------------------
https://www.cnblogs.com/v-July-v/archive/2012/03/22/2413055.html

多看看！

基本思路是： 大数据所以分而治之，然后hash table用的很多，为了统计一些数量，比如统计每个词出现的次数


KMP
------------------
KMP算法易懂版https://www.bilibili.com/video/BV1jb411V78H?from=search&seid=4251637190584004649  
这个视频基本上从原理上讲懂了

::

    def KMP(s, p):
        """
        s为主串
        p为模式串
        如果t里有p，返回打头下标
        """
        nex = getNext(p)
        i = 0
        j = 0  # 分别是s和p的指针
        while i < len(s) and j < len(p):
            if j == -1 or s[i] == p[j]:  # j==-1是由于j=next[j]产生
                i += 1
                j += 1
            else:
                j = nex[j]

        if j == len(p):  # j走到了末尾，说明匹配到了
            return i - j
        else:
            return -1


    def getNext(p):
        """
        p为模式串
        返回next数组，即部分匹配表
        """
        nex = [0] * (len(p) + 1)
        nex[0] = -1
        i = 0
        j = -1
        while i < len(p):
            if j == -1 or p[i] == p[j]:
                i += 1
                j += 1
                nex[i] = j  # 这是最大的不同：记录next[i]
            else:
                j = nex[j]
        return nex

    s="abxababcabyabc"
    p="abyab"
    KMP(s, p)
    
代码贴一下。不是太懂。下面那个getNext(p)只被调用了一次。生成的是一个 list，长度比substring大1，开头是-1.然后似乎是在找重合的前缀后缀。

用pytorch手写逻辑回归
------------------------------------
假设已经准备好数据集了

::

    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    import torch.nn.functional as F

    # 准备数据集和参数
    train_dataset = xxxx
    test_dataset = xxxx

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

    # 核心函数                                  
    class LR(nn.Module):
        def __init__(self,input_dims,output_dims):
            super().__init__()
            self.linear=nn.Linear(input_dims, output_dims,bias=True)
        def forward(self,x):
            x=self.linear(x)
            return x

    model = LR(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

    for epoch in range(num_epochs):
        for i,(images,labels)in enumerate(train_loader):

            # forward
            y_pred = model(images)
            loss = criterion(y_pred,labels)
            
            # backward()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 验证和测试
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28)
            outputs = model(images)
            _ , predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    # 保存模型
    torch.save(LR_model.state_dict(), 'model.ckpt')