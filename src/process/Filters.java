package process;

import GUI.Display;

import javax.imageio.ImageIO;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;
import java.io.File;

/**
 * Created by Dubrzr on 12/07/2016.
 */
public class Filters {
    public static BufferedImage greyscale(BufferedImage bi) {
        ColorSpace cs = ColorSpace.getInstance(ColorSpace.CS_GRAY);
        ColorConvertOp op = new ColorConvertOp(cs, null);
        BufferedImage result = op.filter(bi, null);
        return result;
    }
}
