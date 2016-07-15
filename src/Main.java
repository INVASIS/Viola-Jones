import GUI.Display;
import GUI.ImageHandler;
import cuda.AnyFilter;

/**
 * this is a main class
 */
public class Main {

    public static void main(String[] args) {
        ImageHandler imageHandler = new ImageHandler("data/face.jpg");

        Display.drawImage(imageHandler.getBufferedImage());
        Display.drawImage(imageHandler.getGrayBufferedImage());

        AnyFilter filter = new AnyFilter(imageHandler.getWidth(), imageHandler.getHeight(), imageHandler.getGrayImage());
        filter.conpute();
    }
}
