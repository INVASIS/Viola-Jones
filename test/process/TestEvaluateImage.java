package process;

import org.junit.Assert;
import org.junit.Test;
import process.features.Rectangle;

import java.util.ArrayList;

public class TestEvaluateImage {

    @Test
    public void getAllRectanglesTest() {

        // TODO : maybe verif the rectangles ?

        EvaluateImage evaluateImage = new EvaluateImage(0, 0, null, 19, 19);

        ArrayList<Rectangle> rectangles = evaluateImage.getAllRectangles(19, 19, EvaluateImage.SCALE_COEFF);
        Assert.assertEquals(1, rectangles.size());

        rectangles = evaluateImage.getAllRectangles(19, 18, EvaluateImage.SCALE_COEFF);
        Assert.assertEquals(0, rectangles.size());

        rectangles = evaluateImage.getAllRectangles(19, 20, EvaluateImage.SCALE_COEFF);
        Assert.assertEquals(2, rectangles.size());

        rectangles = evaluateImage.getAllRectangles(22, 22, EvaluateImage.SCALE_COEFF);
        Assert.assertEquals(16, rectangles.size());

        rectangles = evaluateImage.getAllRectangles(23, 22, EvaluateImage.SCALE_COEFF);
        Assert.assertEquals(20, rectangles.size());

        rectangles = evaluateImage.getAllRectangles(23, 23, EvaluateImage.SCALE_COEFF);
        Assert.assertEquals(26, rectangles.size());

        rectangles = evaluateImage.getAllRectangles(23, 23, 1.30f);
        Assert.assertEquals(25, rectangles.size());

    }

}
