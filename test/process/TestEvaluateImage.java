package process;

import org.junit.Assert;
import org.junit.Test;
import process.features.Rectangle;

import java.util.ArrayList;

import static process.ImageEvaluator.getAllRectangles;

public class TestEvaluateImage {

    @Test
    public void getAllRectanglesTest() {
        ArrayList<Rectangle> rectangles = getAllRectangles(19, 19, 1.25f, 1, 1, 1, 19);
        Assert.assertEquals(1, rectangles.size());

        rectangles = getAllRectangles(19, 18, 1.25f, 1, 1, 1, 19);
        Assert.assertEquals(0, rectangles.size());

        rectangles = getAllRectangles(19, 20, 1.25f, 1, 1, 1, 19);
        Assert.assertEquals(2, rectangles.size());

        rectangles = getAllRectangles(22, 22, 1.25f, 1, 1, 1, 19);
        Assert.assertEquals(16, rectangles.size());

        rectangles = getAllRectangles(23, 22, 1.25f, 1, 1, 1, 19);
        Assert.assertEquals(20, rectangles.size());

        rectangles = getAllRectangles(23, 23, 1.25f, 1, 1, 1, 19);
        Assert.assertEquals(26, rectangles.size());

        rectangles = getAllRectangles(23, 23, 1.30f, 1, 1, 1, 19);
        Assert.assertEquals(25, rectangles.size());

    }

}
