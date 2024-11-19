import java.util.*;

public class Practice {
}

//739-m-每日温度
class DailyTemperatures {
    //单调栈
    //递减栈-栈顶元素最小,栈记录的是下标，但是按照温度的规则出入栈
//    public int[] dailyTemperatures(int[] temperatures) {
//        Deque<Integer> stack = new LinkedList<>();
//        int day = 0;
//        int[] res = new int[temperatures.length];
//        //空数组就直接返回
//        if (temperatures == null || temperatures.length == 0) {
//            return null;
//        }
//        //先把0放进栈里
//        stack.push(0);
//        for (int i = 1; i < temperatures.length; i++) {
//            //保持递减栈的特点，遇到栈顶元素要更小就一直出栈，同时更新res数组
//            while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
//                day = stack.pop();
//                res[day] = i - day;
//            }
//            //将该元素下标入栈~
//            stack.push(i);
//        }
//        return res;
//
//    }
//    public int[] dailyTemperatures(int[] temperatures) {
//        int length = temperatures.length;
//        int[] res = new int[length];
//        //单调栈
//        Deque<Integer> stack = new LinkedList<>();
//        //初始化
//        stack.push(0);
//        //开始单调栈更新res
//        for (int i = 1; i < length; i++) {
//            while (!stack.isEmpty() && temperatures[stack.peek()] < temperatures[i]) {//当前温度大于栈内温度
//                int index = stack.pop();
//                res[index] = i - index;
//            }
//            stack.push(i);
//        }
//        return res;
//    }
    public int[] dailyTemperatures(int[] temperatures) {
        Deque<Integer> monoStack = new LinkedList<>();//存index
        int n = temperatures.length;
        int[] answer = new int[n];
        int top = 0;
        // monoStack.push(0);
        for (int i = 0; i < n; i++) {
            while (!monoStack.isEmpty() && temperatures[monoStack.peek()] < temperatures[i]) {
                top = monoStack.pop();
                answer[top] = i - top;
            }
            monoStack.push(i);
        }
        return answer;

    }
}

//221-m-最大正方形
class MaximalSquare {
    //动态规划-dp[i][j]代表以(i,j)为右下角的最大正方形的边长
    public int maximalSquare(char[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }
        int width = matrix[0].length, height = matrix.length;
        int maxr = 0;
        //初始化一下dp左上角的边边上值均为0【初始值】
        int[][] dp = new int[height + 1][width + 1];
        for (int row = 0; row < height; row++) {
            for (int column = 0; column < width; column++) {
                if (matrix[row][column] == '1') {
                    //找到左上角三个部分最短的边长并+1作为自己的边长（当自己的matrix位置值为1时）
                    dp[row + 1][column + 1] = Math.min(Math.min(dp[row][column + 1], dp[row + 1][column]), dp[row][column]) + 1;
                    //顺便找一下最大边长
                    maxr = Math.max(dp[row + 1][column + 1], maxr);
                }
            }
        }
        return maxr * maxr;
    }
}

//208-m-实现前缀树
class Trie {
    boolean isWord;
    Trie[] next;

    public Trie() {
        isWord = false;
        next = new Trie[26];
    }

    public void insert(String word) {
        Trie node = this;
        for (int i = 0; i < word.length(); i++) {
            if (node.next[word.charAt(i) - 'a'] == null) {
                node.next[word.charAt(i) - 'a'] = new Trie();
            }
            node = node.next[word.charAt(i) - 'a'];
        }
        node.isWord = true;

    }

    public boolean search(String word) {
        Trie node = this;
        for (int i = 0; i < word.length(); i++) {
            node = node.next[word.charAt(i) - 'a'];
            if (node == null) {
                return false;
            }
        }
        return node.isWord;
    }

    public boolean startsWith(String prefix) {
        Trie node = this;
        for (int i = 0; i < prefix.length(); i++) {
            node = node.next[prefix.charAt(i) - 'a'];
            if (node == null) {
                return false;
            }
        }
        return true;
    }
}

//207-m-课程表
class CanFinish {
    int[] inDegree;
    List<List<Integer>> edges;//边，记录入度的边；邻接表

    public boolean canFinish(int numCourses, int[][] prerequisites) {
        //初始化入度数组
        inDegree = new int[numCourses];
        //初始化邻接表
        edges = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            edges.add(new ArrayList<>());
        }
        //根据prerequisites更新邻接表edges与入度数组inDegree
        for (int[] info : prerequisites) {
            edges.get(info[1]).add(info[0]);
            inDegree[info[0]]++;
        }

        int num = 0;
        //bfs开始拓扑排序
        Deque<Integer> queue = new LinkedList();
        //入度为0的结点入队
        for (int i = 0; i < numCourses; i++) {
            if (inDegree[i] == 0) {
                queue.offer(i);
            }
        }
        //开始正式bfs
        while (!queue.isEmpty()) {
            int node = queue.poll();
            //记录拓扑排序的个数
            num++;
            for (int i : edges.get(node)) {
                //更新删除节点的入度
                inDegree[i]--;
                //顺便更新一下新加进去的结点（新的入度为0的结点）
                if (inDegree[i] == 0) {
                    queue.offer(i);
                }
            }
        }

        //拓扑排序个数=课程数则无环且可拓
        return num == numCourses;

    }
}

//198-m-打家劫舍:相邻偷窃会报警
class Rob {
    public int rob(int[] nums) {
        //dp[n]代表到第n户最高多少金额
        //dp[n]=max(dp[n-1],dp[n-2]+nums[n-1]);
        int n = nums.length;
        if (n == 0) return 0;
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = nums[0];
        for (int i = 2; i <= n; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i - 1]);
        }
        return dp[n];
    }
}

//169-e-多数元素
class MajorityElement {
    public int majorityElement(int[] nums) {
        Map<Integer, Integer> record = new HashMap<>();
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            record.put(nums[i], record.getOrDefault(nums[i], 0));
            if (i > n / 2) {
                if (record.get(nums[i]) > n / 2) {
                    return nums[i];
                }
            }
        }
        return -1;
    }

    //摩尔投票，更优解法
    public int majorityElement1(int[] nums) {
        int x = 0, votes = 0, count = 0;
        //得到最多数量的x
        //votes为0时改变当前投票，否则遇到相同x则votes+1，不同x则votes-1
        for (int num : nums) {
            if (votes == 0) x = num;
            votes += num == x ? 1 : -1;
        }
        //重新遍历： 验证 x 是否为众数（计算想出现的次数count）
        for (int num : nums)
            if (num == x) count++;
        return count > nums.length / 2 ? x : 0; // 当无众数时返回 0
    }

    //简易排序算法
    public int majorityElement3(int[] nums) {
        Arrays.sort(nums);
        return nums[nums.length / 2];
    }
}

//238-m-除以自身以外数组的乘积
class ProductExceptSelf {
    public int[] productExceptSelf(int[] nums) {
        int length = nums.length;
        int[] left = new int[length];//left[i]指i左侧的乘积
        int[] right = new int[length];//right[i]指i右侧的乘积
        int l = 1, r = 1;
        for (int i = 0; i < length; i++) {
            left[i] = l;
            l *= nums[i];
            right[length - i - 1] = r;
            r *= nums[length - i - 1];
        }

        int[] multiple = new int[length];//multiple[i]=left[i]*right[i]
        for (int i = 0; i < length; i++) {
            multiple[i] = left[i] * right[i];
        }
        return multiple;
    }
}

//155-m-最小栈
//辅助栈
class MinStack {
    Deque<Integer> stack;
    Deque<Integer> minStack;//保存每个stack上的元素对应的最小值

    public MinStack() {
        stack = new LinkedList<>();
        minStack = new LinkedList<>();
        minStack.push(Integer.MAX_VALUE);
    }

    public void push(int val) {
        stack.push(val);
        minStack.push(Math.min(val, minStack.peek()));
    }

    public void pop() {
        stack.pop();
        minStack.pop();
    }

    public int top() {
        return stack.peek();
    }

    public int getMin() {
        return minStack.peek();
    }
}

//152-m-乘积最大子数组
// 动态规划
class MaxProduct {
    public int maxProduct(int[] nums) {
        int max = Integer.MIN_VALUE, imax = 1, imin = 1;
        int temp;
        //imax为当前子数组最大值，imin为当前最小值（作用是遇到负数时交换,最大值变成最小值，最小值变成最大值）
        for (int i = 0; i < nums.length; i++) {
            //遇到负数时交换imin和imax
            if (nums[i] < 0) {
                temp = imax;
                imax = imin;
                imin = temp;
            }
            //关键的一步！！计算当前imax和imin
            imax = Math.max(imax * nums[i], nums[i]);
            imin = Math.min(imin * nums[i], nums[i]);

            //更新max（当前最大子数组）
            max = Math.max(max, imax);
        }
        return max;
    }
}

//146-m-LRU缓存
//双向链表记录lru+map记录key对应的结点
class LRUCache {
    //双向链表增删更好操作
    class BilateralNode {
        int key;
        int value;
        BilateralNode prev;
        BilateralNode next;

        public BilateralNode() {
        }

        public BilateralNode(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }

    BilateralNode head, tail;//保持状态
    Map<Integer, BilateralNode> cache;//记录键值对k-node
    int capacity;
    int length;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        this.length = 0;
        this.cache = new HashMap<>();
        //dummy node--head and tail
        this.head = new BilateralNode();
        this.tail = new BilateralNode();
        head.next = tail;
        tail.prev = head;
    }

    public int get(int key) {
        BilateralNode node = cache.get(key);
        if (node == null) {
            return -1;
        } else {
            moveToHead(node);
            return node.value;
        }
    }

    public void put(int key, int value) {
        BilateralNode node = cache.get(key);
        if (node == null) {
            node = new BilateralNode(key, value);
            addToHead(node);
            cache.put(key, node);
            if (length == capacity) {
                removeTail();
            } else {
                length++;
            }
        } else {
            node.value = value;
            //记住这里更新也需要将node放至头部
            moveToHead(node);
        }

    }

    void addToHead(BilateralNode node) {
        node.next = head.next;
        node.prev = head;
        head.next = node;
        node.next.prev = node;
    }

    void moveToHead(BilateralNode node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
        addToHead(node);
    }

    void removeTail() {
        BilateralNode node = tail.prev;
        node.prev.next = tail;
        tail.prev = node.prev;
        cache.remove(node.key);
    }
}

class ListNode {
    int val;
    ListNode next;

    ListNode(int x) {
        val = x;
        next = null;
    }

    ListNode() {
    }
}

//141-e-环形链表
class HasCycle {
    public boolean hasCycle(ListNode head) {
        if (head == null) {
            return false;
        }
        ListNode fast = head.next, slow = head;
        while (fast != null && fast.next != null) {
            if (fast == slow) {
                return true;
            }
            fast = fast.next.next;
            slow = slow.next;
        }
        return false;
    }
}

//============================w2=============================
//139-m-单词拆分
//动态规划
class WordBreak {
    public boolean wordBreak(String s, List<String> wordDict) {
        //dp[n]:[0,n]的字符串是否能被 wordDict 表示出来
        //dp[n]=dp[j]&&wordDict.contains(s.substring(j,n))
        int len = s.length();
        boolean[] dp = new boolean[len + 1];
        dp[0] = true;
        for (int i = 1; i <= len; i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && wordDict.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[len];
    }
}

//136-e-只出现一次的数字
//数组中所有元素异或所得为该元素，因为其他元素均出现2次
class SingleNumber {
    public int singleNumber(int[] nums) {
        int res = 0;
        for (int num : nums) {
            //异或运算（位运算）
            res ^= num;
        }
        return res;
    }
}

//647-m-找出所有回文串
class CountSubstrings {
    //dp,但有效率更好的中心法
    public int countSubstrings(String s) {
        int length = s.length();
        int count = 0;
        boolean[][] dp = new boolean[length][length];//dp[i][j]代表[i,j]是否为回文串

        for (int j = 0; j < length; j++) {
            for (int i = 0; i <= j; i++) {
                //s[i]=s[j]时，要么i=j要么j=i+1要么其内部的串也是回文串
                if (s.charAt(i) == s.charAt(j) && (j - i < 2 || dp[i + 1][j - 1])) {
                    dp[i][j] = true;
                    count++;
                }
            }

        }
        return count;
    }
}

//28-m-最长连续序列
class LongestConsecutive {
    public int longestConsecutive(int[] nums) {
        //为了好查，先把nums放进set中
        HashSet<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        int maxCount = 0, count = 0, currentNum = 0;
        for (int num : set) {
            //确认为子序列的第一个再做行动
            if (!set.contains(num - 1)) {
                count++;
                currentNum = num + 1;
                //找到num开头的最长子序列
                while (set.contains(currentNum)) {
                    count++;
                    currentNum++;
                }
                maxCount = Math.max(maxCount, count);
                count = 0;
            }
        }
        return maxCount;
    }
}


class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode() {
    }

    TreeNode(int val) {
        this.val = val;
    }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

//124-h-二叉树的最大路径和
//路径是图中的路径：左节点那条最大路+本节点值+右节点最大路
class MaxPathSum {
    int max = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        recursiveSum(root);
        return max;
    }

    int recursiveSum(TreeNode node) {
        if (node == null) return 0;

        //左右子路径如果<0则不要了
        int left = Math.max(recursiveSum(node.left), 0);
        int right = Math.max(recursiveSum(node.right), 0);

        //此处是计算（更新）图意义上的最大路径
        int newPath = left + right + node.val;
        max = Math.max(newPath, max);

        return Math.max(left, right) + node.val;//返回的是本节点的对于二叉树而言的最大路径（而非对于图）
    }
}

//322-m-零钱兑换
class CoinChange {
    public int coinChange(int[] coins, int amount) {
        //dp[n]:n的最小硬币个数
        //dp[n]=min(dp[n],dp[n-coins[i]]+1);
        int[] dp = new int[amount + 1];
        dp[0] = 0;
        int len = coins.length;
        for (int i = 1; i <= amount; i++) {
            dp[i] = Integer.MAX_VALUE;
            for (int coin : coins) {
                if (i - coin >= 0 && dp[i - coin] != Integer.MAX_VALUE) {
                    dp[i] = Math.min(dp[i], dp[i - coin] + 1);
                }
            }
        }
        return dp[amount] == Integer.MAX_VALUE ? -1 : dp[amount];
    }
}

//494-m-目标和
//回溯：较简单理解；也可以用动态规划【复杂暂时不想考虑】
class FindTargetSumWays {
    int count = 0;
    int length = 0;

    public int findTargetSumWays(int[] nums, int target) {
        length = nums.length;
        backTrack(nums, target, 0, 0);
        return count;
    }

    void backTrack(int[] nums, int target, int index, int sum) {
        if (index == length) {
            if (target == sum) count++;
            return;
        }
        backTrack(nums, target, index + 1, sum + nums[index]);
        backTrack(nums, target, index + 1, sum - nums[index]);
    }
}

//461-e-汉明距离
//两个数的二进制不同位的个数
class HammingDistance {
    public int hammingDistance(int x, int y) {
        return Integer.bitCount(x ^ y);//异或得到不同位的表示
    }
}

//448-e-找到数组中所有消失的数字
class FindDisappearedNumbers {
    public List<Integer> findDisappearedNumbers(int[] nums) {
        int x;
        int n = nums.length;
        //原地标记nums
        for (int num : nums) {
            x = (num - 1) % n;//计算num的索引x
            nums[x] += n;//将x处增加n以标记num已经存在
        }

        //计算res数组
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (nums[i] <= n) {
                res.add(i + 1);
            }
        }
        return res;
    }
}

//438-m-找到字符串中所有字母异位词
class FindAnagrams {
    public List<Integer> findAnagrams(String s, String p) {
        //显然是滑动窗口，像之前的那个水果的，但是要找到一个数据结构来存储&判断子串是否为字母异位词
        //对于全是字母的字符串，可以不用HashMap，可以用那个东西、、、数组
        //p为子串
        int sLength = s.length(), pLength = p.length();
        //不合法情况
        List<Integer> res = new ArrayList<>();
        if (pLength > sLength) {
            return res;
        }

        //利用以下的数据结构来存储
        int[] sCount = new int[26];//目前滑动窗口的内容，在改
        int[] pCount = new int[26];//判断基准，不会改

        //存储初识数据
        for (int i = 0; i < pLength; i++) {
            pCount[p.charAt(i) - 'a']++;
            sCount[s.charAt(i) - 'a']++;
        }

        if (Arrays.equals(pCount, sCount)) res.add(0);
        //开始滑动
        for (int i = 0; i < sLength - pLength; i++) {
            //往右滑一步
            sCount[s.charAt(i) - 'a']--;
            sCount[s.charAt(i + pLength) - 'a']++;
            if (Arrays.equals(pCount, sCount)) res.add(i + 1);
        }
        return res;
    }
}

//437-m-路径总和III
class PathSum {
    int res;

    public int pathSum(TreeNode root, int targetSum) {
        res = 0;

        return 0;
    }

    void countPath(TreeNode node, int target) {

    }
}

//160-e-相交链表
class GetIntersectionNode {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode a = headA, b = headB;
        int alength = 0, blength = 0;
        while (a != null) {
            a = a.next;
            alength++;
        }
        while (b != null) {
            b = b.next;
            blength++;
        }
        int dif = Math.abs(alength - blength);
        a = headA;
        b = headB;
        if (alength > blength) {
            for (int i = 0; i < dif; i++) {
                a = a.next;
            }
        } else {
            for (int i = 0; i < dif; i++) {
                b = b.next;
            }
        }
        while (a != null) {
            if (a == b) {
                return a;
            }
            a = a.next;
            b = b.next;
        }
        return null;

    }

    public ListNode getIntersectionNode1(ListNode headA, ListNode headB) {
        ListNode a = headA, b = headB;
        int aLength = 0, bLength = 0;
        while (a != null) {
            a = a.next;
            aLength++;
        }
        while (b != null) {
            b = b.next;
            bLength++;
        }
        int dif = aLength - bLength;
        a = headA;
        b = headB;
        if (dif > 0) {
            while (dif > 0) {
                a = a.next;
                dif--;
            }
        } else {
            while (dif < 0) {
                b = b.next;
                dif++;
            }
        }
        while (a != null && a != b) {
            a = a.next;
            b = b.next;
        }
        return a;
    }
}

//236-m-二叉树的最近共同祖先
class LowestCommonAncestor {
    TreeNode res;

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        res = null;
        existsNode(root, p, q);
        return res;
    }

    boolean existsNode(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return false;
        boolean left = existsNode(root.left, p, q);
        boolean right = existsNode(root.right, p, q);
        if (root == p || root == q || left && right) {
            res = root;//自下往上更新祖先
            return true;
        }
        //返回是否存在p/q
        return left || right;
    }
}

//234-e-回文链表
class IsPalindrome {
    public boolean isPalindrome(ListNode head) {
        List<Integer> list = new ArrayList<Integer>();

        ListNode node = head;
        while (node != null) {
            list.add(node.val);
            node = node.next;
        }
        int front = 0, back = list.size() - 1;
        while (front < back) {
            if (list.get(front) == list.get(back)) {
                front++;
                back--;
            } else {
                return false;
            }
        }
        return true;
    }
}

//283-e-移动零
class MoveZeroes {
    public void moveZeroes(int[] nums) {
        int fast = 0, slow = 0;
        for (fast = 0; fast < nums.length; fast++) {
            if (nums[fast] != 0) {
                nums[slow] = nums[fast];
                slow++;
            }
        }
        while (slow < nums.length) {
            nums[slow++] = 0;
        }
    }
}

//15-m-三数之和
class ThreeSum {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        int i = 0, left = 0, right = nums.length - 1, sum = 0;
        for (; i < nums.length - 2; i++) {
            //i去重
            if (i > 0 && nums[i - 1] == nums[i]) continue;
            left = i + 1;
            right = nums.length - 1;
            while (left < right) {
                sum = nums[i] + nums[left] + nums[right];
                if (sum > 0) {
                    // while(left<right&&nums[right]==nums[right-1])right--;
                    right--;
                } else if (sum < 0) {
                    // while(left<right&&nums[left]==nums[left+1])left++;
                    left++;
                } else {//sum==0
                    res.add(new ArrayList(Arrays.asList(nums[i], nums[left], nums[right])));
                    while (nums[right] == nums[right - 1] && left < right) right--;
                    right--;
                    while (nums[left] == nums[left + 1] && left < right) left++;
                    left++;
                }
            }
        }
        return res;
    }
}

//42-h-接雨水
class Trap {
    public int trap(int[] height) {
        int length = height.length;
        int[] max_left = new int[length];
        int[] max_right = new int[length];
        //update max_left and max_right
        for (int i = 1; i < length; i++) {//从左往右更新
            max_left[i] = Math.max(height[i - 1], max_left[i - 1]);
        }
        for (int i = length - 2; i >= 0; i--) {//从右往左更新
            max_right[i] = Math.max(height[i + 1], max_right[i + 1]);
        }
        //calculate rain
        int res = 0;
        int min = 0;
        for (int i = 0; i < length; i++) {
            min = Math.min(max_left[i], max_right[i]);
            if (min > height[i]) {
                res += min - height[i];
            }
        }
        return res;
    }
}

//3-m-无重复字符的最长子串
class LengthOfLongestSubstring {
    public int lengthOfLongestSubstring(String s) {
        char[] arr = s.toCharArray();
        int left = 0;
        Map<Character, Integer> record = new HashMap<>();
        int max = 0;
        for (int right = 0; right < arr.length; right++) {
            if (record.containsKey(arr[right])) {
                left = Math.max(left, record.get(arr[right]) + 1);
            }
            record.put(arr[right], right);
            max = Math.max(max, right - left + 1);
        }
        return max;
    }
}

//560-m-和为k的子数组
class SubarraySum {
    public int subarraySum(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        int res = 0;
        int pre = 0;
        map.put(0, 1);//这一步很重要，初始化
        for (int i = 0; i < nums.length; i++) {
            pre += nums[i];
            if (map.containsKey(pre - k)) {
                res += map.get(pre - k);
            }
            map.put(pre, map.getOrDefault(pre, 0) + 1);
        }
        return res;
    }
}

//53-m-最大子数组和
class MaxSubArray {
    public int maxSubArray(int[] nums) {
        //dp[i]：以num[i]为结尾的子串
        int[] dp = new int[nums.length];
        int res = nums[0];
        dp[0] = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if (dp[i - 1] < 0) {
                dp[i] = nums[i];
            } else {
                dp[i] = nums[i] + dp[i - 1];
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }
}

//189-m-轮转数组
class Rotate {
    public void rotate(int[] nums, int k) {
        int length = nums.length;
        int[] backup = Arrays.copyOf(nums, length);
        for (int i = 0; i < length; i++) {
            nums[(i + k) % length] = backup[i];
        }
    }
}

//41-h-缺失的第一个正数
class FirstMissingPositive {
    public int firstMissingPositive(int[] nums) {
        int length = nums.length;
        for (int i = 0; i < length; i++) {
            while (nums[i] >= 1 && nums[i] <= length
                    && nums[i] != nums[nums[i] - 1]) {
                swap(nums, nums[i] - 1, i);
            }
        }
        for (int i = 0; i < length; i++) {
            if (i != nums[i] - 1) {
                return i + 1;
            }
        }
        return length + 1;
    }

    void swap(int[] nums, int i1, int i2) {
        int temp = nums[i1];
        nums[i1] = nums[i2];
        nums[i2] = temp;
    }
}

//56-m-合并数组
//蛮难的，主要是一些函数的使用
class Merge {
    public int[][] merge(int[][] intervals) {
        List<int[]> list = new ArrayList<>();
        int L, R;
        if (intervals.length < 1) {
            return null;
        }
        Arrays.sort(intervals,
                (interval1, interval2) -> interval1[0] - interval2[0]);
        for (int i = 0; i < intervals.length; i++) {
            L = intervals[i][0];
            R = intervals[i][1];
            if (list.size() == 0 || list.get(list.size() - 1)[1] < L) {
                //新区间加入list
                list.add(new int[]{L, R});
            } else {
                //需要合并
                list.get(list.size() - 1)[1]
                        = Math.max(list.get(list.size() - 1)[1], R);
            }
        }
        return list.toArray(new int[list.size()][]);
    }
}

//206-e-反转链表
class ReverseList {
    public ListNode reverseList(ListNode head) {
        if (head == null) return null;
        ListNode pre = null, node = head, temp = head.next;
        while (node != null) {
            temp = node.next;
            node.next = pre;

            pre = node;
            node = temp;
        }
        return pre;
    }
}

//142-m-环形链表II
class DetectCycle {
    public ListNode detectCycle(ListNode head) {
        ListNode fast = head, slow = head;
        while (true) {
            if (fast == null || fast.next == null || fast.next.next == null) return null;
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) break;
        }
        fast = head;
        while (fast != slow) {
            fast = fast.next;
            slow = slow.next;
        }
        return fast;
    }
}

//21-e-合并两个有序链表
class MergeTwoLists {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        if (list1 == null) return list2;
        if (list2 == null) return list1;
        ListNode node1 = list1, node2 = list2;
        ListNode list = new ListNode();
        ListNode head = list;
        while (node1 != null && node2 != null) {
            if (node1.val > node2.val) {
                list.next = node2;
                node2 = node2.next;
            } else {
                list.next = node1;
                node1 = node1.next;
            }
            list = list.next;
        }
        if (node1 == null) {
            list.next = node2;
        } else {
            list.next = node1;
        }
        return head.next;
    }
}

//2-m-两数相加
class AddTwoNumbers {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int carry = 0, sum = 0;
        ListNode dummyHead = new ListNode();
        ListNode node = dummyHead;
        int num1 = 0, num2 = 0;
        while (l1 != null || l2 != null) {
            num1 = (l1 != null) ? l1.val : 0;
            num2 = (l2 != null) ? l2.val : 0;
            sum = num1 + num2 + carry;
            carry = sum / 10;
            node.next = new ListNode(sum % 10);
            node = node.next;
            if (l1 != null) l1 = l1.next;
            if (l2 != null) l2 = l2.next;
        }
        if (carry != 0) {
            node.next = new ListNode(carry);
        }
        return dummyHead.next;
    }
}

//19-m-删除链表的倒数第N个结点
class RemoveNthFromEnd {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        //快慢指针
        ListNode dummy = new ListNode();
        dummy.next = head;
        ListNode fast = head, slow = head, pre = dummy;
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }
        while (fast != null) {
            pre = slow;
            slow = slow.next;
            fast = fast.next;
        }
        pre.next = slow.next;
        return dummy.next;
    }
}

//24-m-两两交换链表中的结点
class SwapPairs {
    public ListNode swapPairs(ListNode head) {
        ListNode dummy = new ListNode();
        dummy.next = head;
        ListNode front = dummy, a = head, b = null, back = null;
        while (a != null) {
            b = a.next;
            if (b == null) {
                break;
            }
            back = b.next;

            front.next = b;
            b.next = a;
            a.next = back;

            front = a;
            a = front.next;
        }
        return dummy.next;
    }
}


// Definition for a Node.
class Node {
    int val;
    Node next;
    Node random;

    public Node(int val) {
        this.val = val;
        this.next = null;
        this.random = null;
    }
}

//138-m-随机链表的复制
class CopyRandomList {
    //递归+map==》map是为了存谁已经复制了
    Map<Node, Node> map = new HashMap<>();

    public Node copyRandomList(Node head) {
        if (head == null) return null;
        if (!map.containsKey(head)) {
            Node newNode = new Node(head.val);
            map.put(head, newNode);

            newNode.next = copyRandomList(head.next);
            newNode.random = copyRandomList(head.random);
        }
        return map.get(head);
    }
}

//146-m-LRU缓存
class LRUCache1 {
    int capacity;
    int length;
    Node head;
    Node tail;
    Map<Integer, Node> map;

    public LRUCache1(int capacity) {
        this.capacity = capacity;
        head = new Node();
        tail = new Node();
        head.next = tail;
        tail.pre = head;
        length = 0;
        map = new HashMap<>();
    }

    public int get(int key) {
        if (map.containsKey(key)) {
            Node node = this.map.get(key);
            changeToHead(node);
            return node.val;
        } else {
            return -1;
        }
    }

    public void put(int key, int value) {
        if (map.containsKey(key)) {//update
            Node node = map.get(key);
            node.val = value;
            changeToHead(node);
        } else {//add
            Node node = new Node(key, value);
            this.map.put(key, node);
            addToHead(node);
            this.length++;
            if (length > capacity) {
                removeTail();
                this.length--;
            }
        }
    }

    public void changeToHead(Node node) {
        removeNode(node);
        addToHead(node);
    }

    public void removeNode(Node node) {
        node.pre.next = node.next;
        node.next.pre = node.pre;
    }

    public void addToHead(Node node) {
        node.next = this.head.next;
        this.head.next.pre = node;
        this.head.next = node;
        node.pre = this.head;
    }

    public void removeTail() {
        Node node = this.tail.pre;
        removeNode(node);
        this.map.remove(node.key);
    }


    class Node {
        int key;
        int val;
        Node pre;
        Node next;

        Node(int key, int val) {
            this.key = key;
            this.val = val;
            this.pre = null;
            this.next = null;
        }

        Node() {
            this.pre = null;
            this.next = null;
        }
    }
}

//148-m-排序链表
class SortList {
    public ListNode sortList(ListNode head) {
        //终止条件
        if (head == null || head.next == null) return head;
        //找中点
        ListNode fast = head.next, slow = head;//注意初始条件！！如果快慢初始相等，那两个结点的链表永远都分不开，就会永远递归
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode mid = slow.next;//中点
        slow.next = null;//断开链表
        //递归
        ListNode left = sortList(head);
        ListNode right = sortList(mid);
        //合并两个有序链表
        ListNode dummy = new ListNode();//头结点
        ListNode node = dummy;
        while (left != null && right != null) {
            if (left.val < right.val) {
                node.next = left;
                node = left;
                left = left.next;
            } else {
                node.next = right;
                node = right;
                right = right.next;
            }
        }
        if (left != null) {
            node.next = left;
        } else {
            node.next = right;
        }
        return dummy.next;
    }
}

//94-e-二叉树的中序遍历
class InorderTraversal {
    List<Integer> res = new ArrayList<>();

    public List<Integer> inorderTraversal(TreeNode root) {
        inorder(root);
        return res;
    }

    void inorder(TreeNode node) {
        if (node == null) return;
        inorder(node.left);
        res.add(node.val);
        inorder(node.right);
    }
}

//104-e-二叉树的最大深度
class MaxDepth {
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        int left = maxDepth(root.left);
        int right = maxDepth(root.right);
        return Math.max(left, right) + 1;
    }
}

////226-e-翻转二叉树
class InvertTree {
    public TreeNode invertTree(TreeNode root) {
        if (root == null) return root;
        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);
        root.left = right;
        root.right = left;
        return root;
    }
}

//101-e-对称二叉树
class IsSymmetric {
    public boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        return tell(root.left, root.right);
    }

    boolean tell(TreeNode left, TreeNode right) {
        if (left == null && right == null) return true;
        if (left == null || right == null) return false;
        if (left.val != right.val) return false;
        boolean inner = tell(left.right, right.left);
        boolean outer = tell(left.left, right.right);
        return inner && outer;

    }
}

//543-e-二叉树的直径
class DiameterOfBinaryTree {
    int res = 0;

    public int diameterOfBinaryTree(TreeNode root) {
        longerHeight(root);
        return res;
    }

    int longerHeight(TreeNode node) {
        if (node == null) return 0;
        int left = longerHeight(node.left);
        int right = longerHeight(node.right);
        //更新res
        res = Math.max(left + right, res);
        //return
        return Math.max(left, right) + 1;
    }
}

//102-m-层序遍历
class LevelOrder {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        Deque<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int size = 0;
        while (!queue.isEmpty()) {
            size = queue.size();
            List<Integer> list = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                list.add(node.val);
                //queue绝对不可以存在null
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            res.add(list);
        }
        return res;
    }
}

//108-e-将有序数组转换为二叉搜索树
class SortedArrayToBST {
    public TreeNode sortedArrayToBST(int[] nums) {
        if (nums.length == 0) return null;
        return generate(nums, 0, nums.length - 1);
    }

    TreeNode generate(int[] nums, int leftIndex, int rightIndex) {
        if (leftIndex > rightIndex) return null;
        if (leftIndex == rightIndex) return new TreeNode(nums[leftIndex]);
        int mid = (leftIndex + rightIndex) / 2;
        TreeNode left = generate(nums, leftIndex, mid - 1);
        TreeNode right = generate(nums, mid + 1, rightIndex);
        TreeNode node = new TreeNode(nums[mid], left, right);
        return node;
    }
}

//98-m-验证二叉搜索树
class IsValidBST {
    public boolean isValidBST(TreeNode root) {
        return isValid(root, null, null);
    }

    boolean isValid(TreeNode node, Integer lower, Integer higher) {
        if (node == null) return true;
        int val = node.val;
        if (lower != null && val <= lower) return false;
        if (higher != null && val >= higher) return false;

        boolean left = isValid(node.left, lower, val);
        boolean right = isValid(node.right, val, higher);

        return left && right;
    }
}

//230-m-二叉搜索树第k小的元素
class KthSmallest {
    int count = 0;
    int res = 0;

    public int kthSmallest(TreeNode root, int k) {
        count = k;
        find(root, k);
        return res;
    }

    void find(TreeNode node, int k) {
        if (node == null) return;
        find(node.left, k);
        count--;
        if (count == 0) {
            res = node.val;
        }
        if (count < 0) {
            return;
        }
        find(node.right, k);
    }
}

//199-m-二叉树的右视图
class RightSideView {
    public List<Integer> rightSideView(TreeNode root) {
        //层序遍历
        List<Integer> res = new ArrayList<>();
        Deque<TreeNode> queue = new LinkedList<>();
        if (root == null) return res;
        queue.offer(root);
        int length = 0;
        TreeNode node = null;

        while (!queue.isEmpty()) {
            length = queue.size();
            for (int i = 0; i < length; i++) {
                node = queue.poll();
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
                if (i == length - 1) {
                    res.add(node.val);
                }
            }
        }
        return res;
    }
}

//114-m-二叉树展开为链表
class Flatten {
    TreeNode dummy = new TreeNode();
    TreeNode node = dummy;

    public void flatten(TreeNode root) {
        flat(root);
        dummy.right = null;
    }

    void flat(TreeNode root) {
        if (root == null) return;
        TreeNode left = root.left;
        TreeNode right = root.right;
        root.left = null;
        node.right = root;
        node = node.right;
        flat(left);
        flat(right);
    }
}

//105-m-从前序与中序遍历构造二叉树
class BuildTree {
    Map<Integer, Integer> inIndex;

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        inIndex = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            inIndex.put(inorder[i], i);
        }
        int size = inorder.length;
        return buildNode(preorder, inorder, 0, size - 1, 0, size - 1);
    }

    TreeNode buildNode(int[] preorder, int[] inorder,
                       int pF, int pB,
                       int iF, int iB) {
        if (pF > pB || iF > iB) return null;
        TreeNode node = new TreeNode(preorder[pF]);
        int iMid = inIndex.get(preorder[pF]);
        int lsize = iMid - iF;
        int rsize = iB - iMid;
        node.left = buildNode(preorder, inorder, pF + 1, pF + lsize, iF, iMid - 1);
        node.right = buildNode(preorder, inorder, pF + 1 + lsize, pB, iMid + 1, iB);
        return node;
    }
}


//437-m-路径总和III
//long targetSum记得改
class PathSum3 {
    public int pathSum(TreeNode root, long targetSum) {
        if (root == null) return 0;
        int count = 0;
        count += nodeSum(root, targetSum);
        count += pathSum(root.left, targetSum);
        count += pathSum(root.right, targetSum);
        return count;
    }

    int nodeSum(TreeNode node, long targetSum) {//以node为起点的路径有多少符合条件的
        if (node == null) return 0;
        int count = 0;
        if (node.val == targetSum) count++;//node是否为targetSum
        count += nodeSum(node.left, targetSum - node.val);
        count += nodeSum(node.right, targetSum - node.val);
        return count;
    }
}

//200-m-岛屿数量
class NumIslands {
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0) {
            return 0;
        }
        int res = 0;
        for (int x = 0; x < grid.length; x++) {
            for (int y = 0; y < grid[0].length; y++) {
                if (grid[x][y] == '1') {
                    if (isIsland(grid, x, y)) {
                        res++;
                    }
                }
            }
        }
        return res;
    }

    boolean isIsland(char[][] grid, int x, int y) {
        if (outOfBounds(grid, x, y)) return false;
        if (grid[x][y] == '1') {
            grid[x][y] = '2';
            isIsland(grid, x + 1, y);
            isIsland(grid, x - 1, y);
            isIsland(grid, x, y + 1);
            isIsland(grid, x, y - 1);
            return true;
        }
        return false;
    }

    boolean outOfBounds(char[][] grid, int x, int y) {
        if (x >= 0 && x < grid.length
                && y >= 0 && y < grid[0].length) {
            return false;
        }
        return true;
    }
}

//994-m-腐烂的橘子
class OrangesRotting {
    Deque<int[]> rottenQ;
    int isRotted;
    int isFresh;

    public int orangesRotting(int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0) return 0;
        rottenQ = new LinkedList<>();
        isFresh = 0;
        isRotted = 0;
        //双重遍历更新rottenQ,isRotted,isFresh
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    isFresh++;
                }
                if (grid[i][j] == 2) {
                    isRotted++;
                    rottenQ.offer(new int[]{i, j});
                }
            }
        }
        //利用queue开始层序遍历
        int minites = 0;
        if (isFresh == 0) {
            return minites;
        }
        while (rottenQ.size() != 0) {
            if (isFresh == 0) return minites;
            int length = rottenQ.size();
            for (int i = 0; i < length; i++) {
                int[] orange = rottenQ.poll();
                int x = orange[0];
                int y = orange[1];
                isFresh -= rotting(grid, x + 1, y);
                isFresh -= rotting(grid, x - 1, y);
                isFresh -= rotting(grid, x, y + 1);
                isFresh -= rotting(grid, x, y - 1);
            }
            minites++;
        }
        return isFresh == 0 ? minites : -1;
    }

    int rotting(int[][] grid, int x, int y) {
        if (outOfBounds(grid, x, y)) {
            return 0;
        }
        if (grid[x][y] == 1) {
            grid[x][y] = 2;//腐烂
            rottenQ.offer(new int[]{x, y});
            return 1;
        }
        return 0;
    }

    boolean outOfBounds(int[][] grid, int x, int y) {
        if (x >= 0 && x < grid.length && y >= 0 && y < grid[0].length) {
            return false;
        }
        return true;
    }
}

//35-e-搜索插入位置
class SearchInsert {
    public int searchInsert(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        int mid = 0;
        while (left <= right) {
            mid = (left + right) / 2;
            if (nums[mid] == target) return mid;
            if (nums[mid] > target) {
                right = mid - 1;
                continue;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }
}

//74-m-搜索二维矩阵
class SearchMatrix {
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) return false;
        int m = matrix.length;
        int n = matrix[0].length;
        int left = 0;
        int right = m * n - 1;
        int x = 0, y = 0, mid = 0;
        while (left <= right) {
            mid = (left + right) / 2;
            x = mid / n;
            y = mid % n;
            if (matrix[x][y] == target) return true;
            if (matrix[x][y] > target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return false;
    }
}

//34-m-在排序数组中查找元素的第一个和最后一个位置
class SearchRange {
    public int[] searchRange(int[] nums, int target) {
        int[] res = new int[]{-1, -1};
        if (nums == null || nums.length == 0) {
            return res;
        }
        if (nums.length == 1 && target == nums[0]) {
            return new int[]{0, 0};
        }
        int left = 0, right = nums.length - 1;
        int mid = 0;
        while (left <= right) {
            mid = (left + right) / 2;
            if (nums[mid] == target) break;
            if (nums[mid] > target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        if (nums[mid] != target) return res;

        left = mid;
        right = mid;
        while ((left >= 0 && right <= nums.length - 1) && (nums[left] == target || nums[right] == target)) {
            if (nums[left] == target) {
                left--;
            }
            if (nums[right] == target) {
                right++;
            }
        }
        if (left == -1) left = 0;
        if (right == nums.length) right = nums.length - 1;
        if (nums[left] != target) left++;
        if (nums[right] != target) right--;
        res[0] = left;
        res[1] = right;
        return res;
    }
}

//33-m-搜索旋转排序数组
class Search {
    public int search(int[] nums, int target) {
        if (nums == null || nums.length == 0) return -1;
        int left = 0, right = nums.length - 1;
        int mid = 0;
        while (left <= right) {
            //一边有序 一边无序
            mid = (left + right) / 2;
            if (nums[mid] == target) return mid;

            if (nums[left] <= nums[mid]) {//左边有序,要注意这里必须有个=
                if (target < nums[left] || target > nums[mid]) {//target不在左边
                    left = mid + 1;
                } else {//target在左边
                    right = mid - 1;
                }
            } else {//右边有序
                if (target < nums[mid] || target > nums[right]) {//target不在右边
                    right = mid - 1;
                } else {//target在右边
                    left = mid + 1;
                }
            }
        }
        return -1;

    }
}

//153-m-寻找旋转排序数组中的最小值
class FindMin {
    public int findMin(int[] nums) {
        int left = 0, right = nums.length - 1;
        int mid = 0;
        //没旋转
        if (nums[0] <= nums[right]) return nums[0];
        //旋转了，找谷底
        while (left <= right) {
            mid = (left + right) / 2;
            if (nums[mid] >= nums[0]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return nums[left];
    }
}

//46-m-全排列
class Permute {
    List<List<Integer>> res;

    public List<List<Integer>> permute(int[] nums) {
        res = new ArrayList<>();
        int n = nums.length;
        dfs(n, 0, new ArrayList<>(), new boolean[n], nums);
        return res;

    }

    void dfs(int len, int depth, List<Integer> path, boolean[] used, int[] nums) {
        if (depth == len) {
            res.add(new ArrayList<>(path));
            return;
        }

        for (int i = 0; i < len; i++) {
            if (!used[i]) {
                path.add(nums[i]);
                used[i] = true;
                dfs(len, depth + 1, path, used, nums);
                used[i] = false;
                path.remove(path.size() - 1);
            }
        }
    }
}

//70-e-爬楼梯
class ClimbStairs {
    public int climbStairs(int n) {
        if (n == 0) return 1;
        if (n == 1) return 1;
        if (n == 2) return 2;
        int[] dp = new int[n + 1];
        // dp[n]:n阶有多少种方法
        //dp[n]=dp[n-1]+dp[n-2]
        dp[0] = 1;
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i < n + 1; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }
}

//118-e-杨辉三角
class Generate {
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> row = new ArrayList<>();
        row.add(1);
        if (numRows == 0) return res;
        res.add(row);
        if (numRows == 1) return res;
        for (int i = 2; i <= numRows; i++) {
            row = new ArrayList<>();
            row.add(1);
            for (int j = 1; j < i - 1; j++) {
                row.add(res.get(i - 2).get(j) + res.get(i - 2).get(j - 1));
            }
            row.add(1);
            res.add(row);
        }
        return res;
    }
}

//279-m-完全平方数
class NumSquares {
    public int numSquares(int n) {
        //dp[n]:n的完全平方数的最少数量
        //dp[n]=min(dp[n],dp[n-i*i]+1) where i \in (0,根号n)
        //需要循环遍历得到dp[n]；因此需要双重循环
        if (n == 0) return 0;
        if (n == 1) return 1;
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = Integer.MAX_VALUE;
            for (int j = 1; j * j <= i; j++) {
                dp[i] = Math.min(dp[i], dp[i - j * j] + 1);
            }
        }
        return dp[n];
    }
}

//300-m-最长递增子序列
class lengthOfLIS {
    public int lengthOfLIS(int[] nums) {
        //dp[n]:以nums[n]为结尾的最长严格递增子序列的长度
        //dp[n]=max(dp[n],dp[j]+1) where j<n && nums[j]<nums[n]
        //dp[0…i−1] 中最长的上升子序列后面再加一个 nums[i]
        int n = nums.length;
        if (n == 0) return 0;
        int[] dp = new int[n];
        int res = 1;
        dp[0] = 1;
        for (int i = 1; i < n; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }
}

//25-h-k个一组翻转链表
class ReverseKGroup {
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummy = new ListNode();
        ListNode begin = dummy, over = null;
        begin.next = head;
        ListNode start = head, end = head;
        while (start != null) {
            start = begin.next;
            end = start;
            for (int i = 0; i < k - 1; i++) {
                if (end == null) return dummy.next;
                end = end.next;
            }
            if (end == null) return dummy.next;
            over = end.next;
            end.next = null;
            begin.next = reverse(start, end);
            start.next = over;
            begin = start;
        }
        return dummy.next;
    }

    ListNode reverse(ListNode start, ListNode end) {
        ListNode node = start;
        ListNode pre = null;
        ListNode tempt = null;
        while (node != null) {
            tempt = node.next;
            node.next = pre;
            pre = node;
            node = tempt;
        }
        return end;
    }
}

//23-h-合并k个升序链表
class MergeKLists {
    public ListNode mergeKLists(ListNode[] lists) {
        int length = lists.length;
        if (length == 0) return null;
        if (length == 1) return lists[0];
        ListNode dummy = new ListNode();
        for (int i = 0; i < length; i++) {
            dummy.next = mergeTwo(dummy.next, lists[i]);
        }
        return dummy.next;
    }

    ListNode mergeTwo(ListNode head1, ListNode head2) {
        ListNode dummy = new ListNode();
        ListNode node1 = head1, node2 = head2, node = dummy;
        while (node1 != null || node2 != null) {
            if (node1 == null) {
                node.next = node2;
                node2 = node2.next;
            } else if (node2 == null) {
                node.next = node1;
                node1 = node1.next;
            } else {
                if (node1.val < node2.val) {
                    node.next = node1;
                    node1 = node1.next;
                } else {
                    node.next = node2;
                    node2 = node2.next;
                }
            }
            node = node.next;
        }
        return dummy.next;
    }
}

//121-e-买卖股票的最佳时机
class MaxProfit {
    public int maxProfit(int[] prices) {
        if (prices.length <= 1) return 0;
        int minPrice = Integer.MAX_VALUE, res = prices[1] - prices[0];
        for (int i = 0; i < prices.length; i++) {
            if (prices[i] < minPrice) {
                minPrice = prices[i];
            } else {
                if (res < (prices[i] - minPrice)) {
                    res = prices[i] - minPrice;
                }
            }
        }
        return res > 0 ? res : 0;
    }
}

//55-m-跳跃游戏
class CanJump {
    public boolean canJump(int[] nums) {
        int max = 0;
        int len = nums.length;
        for (int i = 0; i < len; i++) {
            if (max >= i) {
                max = Math.max(max, i + nums[i]);
                if (max >= len - 1) {
                    return true;
                }
            }
        }
        return false;
    }
}

//45-m-跳跃游戏II
class Jump {
    public int jump(int[] nums) {
        int max = 0;
        int end = 0;
        int steps = 0;
        int len = nums.length;
        for (int i = 0; i < len - 1; i++) {//不需要考虑最后一个位置的跳跃问题
            max = Math.max(max, i + nums[i]);//持续更新最远位置
            if (i == end) {//走到边界（需要跳跃的位置）
                end = max;//将边界更新到之前找到的最远距离
                steps++;//跳跃，但你不知道是跳跃到哪里
            }
        }
        return steps;
    }
}

//763-m-划分字母区间
class PartitionLabels {
    public List<Integer> partitionLabels(String s) {
        int len = s.length();
        int[] last = new int[26];
        for (int i = 0; i < len; i++) {
            last[s.charAt(i) - 'a'] = Math.max(last[s.charAt(i) - 'a'], i);
        }

        int start = 0, end = 0;
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < len; i++) {
            end = Math.max(end, last[s.charAt(i) - 'a']);
            if (i == end) {
                res.add(end - start + 1);
                start = end + 1;
            }
        }
        return res;
    }
}

//416-m-分割等和子集
class CanPartition {
    public boolean canPartition(int[] nums) {
        //dp[i][j]:前i个是否可以得到总和为j
        int n = nums.length;
        if (n < 2) return false;
        int max = 0, sum = 0;
        for (int i = 0; i < n; i++) {
            sum += nums[i];
            max = Math.max(nums[i], max);
        }
        if (sum % 2 == 1) return false;
        int half = sum / 2;
        if (max > half) return false;//全是正整数
        //准备dp
        boolean[][] dp = new boolean[n][half + 1];
        //dp[][0]填true
        for (int i = 0; i < n; i++) {
            dp[i][0] = true;
        }
        //dp
        for (int i = 1; i < n; i++) {
            for (int j = 1; j <= half; j++) {
                if (j >= nums[i]) {
                    dp[i][j] = dp[i - 1][j] | dp[i - 1][j - nums[i]];
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[n - 1][half];
    }
}

//62-m-不同路径
class UniquePaths {
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        //初始化，左上边界都为1
        for (int i = 0; i < m; i++) dp[i][0] = 1;//左边界
        for (int i = 0; i < n; i++) dp[0][i] = 1;//上边界
        //dp[i][j]=dp[i-1][j]+dp[i][j-1];
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }
}

//64-m-最小路径和
class MinPathSum {
    public int minPathSum(int[][] grid) {
        //dp[i][j]:从(0,0)到(i,j)的最小路径和
        int[][] dp = new int[grid.length][grid[0].length];
        for (int i = 0; i < dp.length; i++) {
            for (int j = 0; j < dp[0].length; j++) {
                if (i == 0 && j == 0) {//左上边界
                    dp[i][j] = grid[i][j];
                } else if (i == 0) {//左边界
                    dp[i][j] = dp[i][j - 1] + grid[i][j];
                } else if (j == 0) {//上边界
                    dp[i][j] = dp[i - 1][j] + grid[i][j];
                } else {//不在边界上
                    dp[i][j] = Math.min(dp[i][j - 1], dp[i - 1][j]) + grid[i][j];
                }
            }
        }
        return dp[dp.length - 1][dp[0].length - 1];
    }
}

//1143-m-最长公共子序列
class LongestCommonSubsequence {
    public int longestCommonSubsequence(String text1, String text2) {
        //dp[i][j]:text1[0,i-1]和text2[0,j-1]的最长公共子序列
        int m = text1.length();
        int n = text2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            char c1 = text1.charAt(i - 1);
            for (int j = 1; j <= n; j++) {
                char c2 = text2.charAt(j - 1);
                if (c1 == c2) {//新的一个相等字符
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m][n];
    }
}

//72-m-编辑距离
class MinDistance {
    public int minDistance(String word1, String word2) {
        //dp[i][j]:word1[0,i-1]和word2[0,j-1]的最小编辑距离
        int m = word1.length();
        int n = word2.length();
        if (m * n == 0) {
            return m + n;
        }
        int[][] dp = new int[m + 1][n + 1];
        //初始化:空子串到任何子串的编辑距离
        for (int i = 0; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j <= n; j++) {
            dp[0][j] = j;
        }
        //dp
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                int left = dp[i - 1][j] + 1;//插入字符
                int up = dp[i][j - 1] + 1;//插入字符
                int both = dp[i - 1][j - 1];
                if (word1.charAt(i - 1) != word2.charAt(j - 1)) {
                    both += 1;//修改字符
                }
                dp[i][j] = Math.min(both, Math.min(left, up));
            }
        }
        return dp[m][n];
    }
}

//32-h-最长有效括号
class LongestValidParentheses {
    public int longestValidParentheses(String s) {
        //dp[i]:以s[i]为结尾的最长有效括号长度
        int n = s.length();
        int[] dp = new int[n];
        int max = 0;
        for (int i = 1; i < n; i++) {
            if (s.charAt(i) == ')') {//某对括号的结尾
                if (s.charAt(i - 1) == '(') {
                    dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;//这对括号的前部分就在旁边
                } else if (i >= dp[i - 1] + 1 && s.charAt(i - dp[i - 1] - 1) == '(') {//可能在上一对之前
                    dp[i] = dp[i - 1] + 2 + (i >= dp[i - 1] + 2 ? dp[i - dp[i - 1] - 2] : 0);//上一对+外面这对+外面这对左边的那一对（因为可能连上了）
                }
            }
            max = Math.max(max, dp[i]);
        }
        return max;
    }
}

//20-e-有效的括号
class IsValid {
    public boolean isValid(String s) {
        int len = s.length();
        if (len % 2 == 1) return false;
        Map<Character, Character> map = new HashMap<>();
        map.put(')', '(');
        map.put(']', '[');
        map.put('}', '{');

        Deque<Character> stack = new LinkedList<>();
        for (int i = 0; i < len; i++) {
            char c = s.charAt(i);
            if (map.containsKey(c)) {
                if (map.isEmpty() || stack.peek() != map.get(c)) {
                    return false;
                }
                stack.pop();
            } else {
                stack.push(c);
            }
        }
        return stack.isEmpty();
    }
}

//394-m-字符串解码
class DecodeString {
    public String decodeString(String s) {
        int k = 0;//当前括号内的k值
        Deque<StringBuilder> resStack = new LinkedList<>();//放的是外层括号内部需要拼接的字符串
        Deque<Integer> kStack = new LinkedList<>();//栈顶放的是目前所在的括号所需要重复的次数（外部括号记录到的k值）
        StringBuilder res = new StringBuilder();//目前括号内部的字符串
        int len = s.length();
        for (int i = 0; i < len; i++) {
            char c = s.charAt(i);
            if (c == '[') {//k和res入栈，k和res重置
                kStack.push(k);
                resStack.push(res);
                k = 0;
                res = new StringBuilder();
            } else if (c == ']') {//解码目前部分的字符串
                StringBuilder temp = new StringBuilder();//重复字符串
                // k=kStack.pop();//这时候不存在k被kstack覆盖的情况
                int curk = kStack.pop();//这块要注意一下
                for (int j = 0; j < curk; j++) temp.append(res);
                res = resStack.pop().append(temp);//合并
            } else if (c >= 'a' && c <= 'z') {//遇到字母直接附加到res中
                res.append(c);
            } else {//数字
                k = k * 10 + (c - '0');
            }
        }
        return res.toString();
    }
}