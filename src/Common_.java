import java.util.*;

public class Common_ {
}

class Map_ {
    void method() {
        int key = 0, value = 0;
        int DEFAULT_VALUE = 0;
        HashMap<Integer, Integer> record = new HashMap<>();
        //判定k/v存在
        record.containsKey(key);
        record.containsValue(value);
        //读取与存入
        record.put(key, value);//存入，写
        record.get(key);//读取value
        record.getOrDefault(key, DEFAULT_VALUE);//读取key对应value或默认值DEFAULT_VALUE；经常与put一起使用
        //遍历
        record.values();//value的集合
        //删除
        record.remove(key);
    }
}

class Set_ {
    void method() {
        int item = 0;
        HashSet<Integer> hashSet = new HashSet<>();
        hashSet.contains(item);
        hashSet.add(item);
        hashSet.remove(item);
        hashSet.clear();

        //遍历
        for (int element : hashSet) {
            //do something...
        }
    }
}

class String_and_CharArray {
    void method() {
        //字符串与字符数组
        String s = "hello";
        //相互转换
        char[] cArr = s.toCharArray();
        String str = new String(cArr);

        //字符串切割
        int begin = 0;
        int end = 1;
        String substring = str.substring(begin, end);//[begin, end)

        //StringBuilder
        StringBuilder sb = new StringBuilder();
        String sb2s = sb.toString();
        sb.append("hi");//StringBuilder拼接
    }
}

class Array_ {
    void method() {
        //数组常用
        int[] array = new int[6];
        int[] array1 = new int[6];
        int[][] intervals = new int[6][];
        Arrays.sort(array);
//        Arrays.asList()
        Arrays.equals(array, array1);
//        Arrays.copyOf();
        Arrays.sort(intervals, (interval1, interval2) -> interval1[0] - interval2[0]);//按顺序就是默认递增
//        Arrays.sort(intervals, (interval1, interval2) -> {return interval1[0] - interval2[0];});//按顺序就是默认递增
//        array.indexOf()不存在这个

    }
}

class List_ {
    void method() {
        //list is abstract; create with ArrayList/LinkedList
        int item = 0;
        int index = 1;
        List<Integer> list = new ArrayList<>();
        list.add(item);
        list.get(index);
        list.contains(item);
        //创建一个有内容的list
        new ArrayList<Integer>(Arrays.asList(1, 2, 3));
        //list转化为数组
        list.toArray(new Integer[list.size()]);//返回Integer[]无法自动装箱
        //队列
        Deque<Integer> queue = new LinkedList<>();
        queue.offer(item);//队尾
        queue.poll();//队头
        queue.peek();//队头
        queue.isEmpty();
        queue.size();
    }
}

class StackAndQueue {
    void method() {
        int item = 0;
        int index = 1;
        //队列Queue
        Deque<Integer> queue = new LinkedList<>();
        queue.offer(item);//队尾
        queue.poll();//队头
        queue.peek();//队头
        queue.isEmpty();
        queue.size();
        //stack
        Deque<Integer> stack = new LinkedList<>();
        stack.push(item);
        stack.pop();
        stack.peek();//stack的头
    }
}

class Trie1 {
    boolean isWord;
    Trie1[] next;

    public Trie1() {
        isWord = false;
        next = new Trie1[26];//注意一下这个构造器的写法
    }
}


