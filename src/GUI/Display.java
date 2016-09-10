package GUI;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.TimerTask;

public class Display {

    public static void drawImage(BufferedImage bi, int timestamp) {

        ImageIO.setUseCache(false);
        JFrame jframe = new JFrame("Image display");
        jframe.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        ImagePanel vidpanel = new ImagePanel(jframe);
        jframe.setContentPane(vidpanel);
        jframe.setSize(640, 480);
        jframe.setVisible(true);
        vidpanel.setImage(bi);
        vidpanel.repaint();
        if (timestamp > 0) {
            new java.util.Timer().schedule(
                    new java.util.TimerTask() {
                        @Override
                        public void run() {
                            jframe.dispose();
                        }
                    },
                    timestamp
            );
        }
    }

    public static void drawImage(BufferedImage bi) {
        drawImage(bi, -1);

    }

    public static void drawImageFromFile(String pathToImage) {
        try {
            BufferedImage bi = ImageIO.read(new File(pathToImage));
            Display.drawImage(bi);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
