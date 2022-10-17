sent1 = """
import java.util.Scanner;

public class HelloWorld {

    public static void main(String[] args) {
        Scanner reader = new Scanner(System.in);
        System.out.print("Enter a number: ");

        // nextInt() reads the next integer from the keyboard
        int number = new Scanner(System.in).nextInt();

        // println() prints the following line to the output screen
        System.out.println("You entered: " + number);
    }
}
"""
sent2 = """
import java.util.Scanner;

public class HiWorld {

    public static void main(String[] args) {

        // Creates a reader instance which takes
        // input from standard input - keyboard
        Scanner reader = new Scanner(System.in);
        System.out.print("Enter a number: ");

        // nextInt() reads the next integer from the keyboard
        int number = reader.nextInt();

    }
}
"""
example = {"sent1": sent1, "sent2": sent2}
