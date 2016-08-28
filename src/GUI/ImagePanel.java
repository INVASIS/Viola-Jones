package GUI;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;


public class ImagePanel extends JPanel {
    public BufferedImage image;
    private JFrame jFrame;

    /**
     * Create the ImagePanel
     *
     * @param width  the width of the image
     * @param height the height of the image
     */
    public ImagePanel(int width, int height) {
        image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
    }

    /**
     * Create an empty ImagePanel
     */
    public ImagePanel(JFrame jFrame) {
        this(1, 1);
        this.jFrame = jFrame;
    }

    /**
     * Create the ImagePanel
     *
     * @param image: image to display
     * @param name:  name of the image
     */
    public ImagePanel(BufferedImage image, String name) {
        this.image = image;
    }

    /**
     * Create the ImagePanel
     *
     * @param file: image to display
     */
    public ImagePanel(File file) {
        try {
            image = ImageIO.read(file);
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public int getWidth() {
        Insets insets = jFrame.getInsets();
        return jFrame.getWidth() - insets.left - insets.right;
    }

    @Override
    public int getHeight() {
        Insets insets = jFrame.getInsets();
        return jFrame.getHeight() - insets.top - insets.bottom;
    }

    public BufferedImage getImage() {
        return image;
    }

    public void setImage(BufferedImage image) {
        this.image = image;
    }

    @Override
    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        g.drawImage(image, 0, 0, getWidth(), getHeight(), null);
    }

}
