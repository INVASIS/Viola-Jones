package GUI;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class Display {

    public static void drawImage(String pathToImage) {


        ImageIO.setUseCache(false);
        JFrame jframe = new JFrame("Image display");
        jframe.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        ImagePanel vidpanel = new ImagePanel(jframe);
        jframe.setContentPane(vidpanel);
        jframe.setSize(640, 480);
        jframe.setVisible(true);

        try {
            BufferedImage bi = ImageIO.read(new File(pathToImage));
            vidpanel.setImage(bi);
            vidpanel.repaint();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
