import GUI.Display;
import GUI.ImageHandler;

/**
 * this is a main class
 */
public class Main {

    public static void main(String[] args) {
        ImageHandler imageHandler = new ImageHandler("data/face.jpg");
        Display.drawImage(imageHandler.getBufferedImage());
        Display.drawImage(imageHandler.getGrayBufferedImage());
    }
}
