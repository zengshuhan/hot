public class Test {
    public static void main(String[] args) {
        String s = "hello";
        //字符串切割
        int begin = 0;
        int end = 1;
        String substring = s.substring(begin, end);//[begin, end)
        System.out.println(substring);
    }
}
