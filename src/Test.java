import java.util.Hashtable;

public class Test {
    public static void main(String[] args) {
        String s = "hello";
        //字符串切割
        int begin = 0;
        int end = 1;
        String substring = s.substring(begin, end);//[begin, end)
//        System.out.println(substring);
        System.out.println(s.charAt(0));//这个charAt是从左往右处理的

        Hashtable hashtable = new Hashtable();
    }

}
