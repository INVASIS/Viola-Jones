package cuda;

import GUI.ImageHandler;
import org.junit.Assert;
import org.junit.Test;
import process.Conf;

import java.util.List;

public class TestHaarExtractor {

    @Test
    public void testComputeFeatures() {

        if (Conf.USE_CUDA) {

            Conf.haarExtractor.setUp(19, 19);
            ImageHandler imageHandler = new ImageHandler("data/testset-19x19/face-png/face00001.png");
            int c = 0;
            for (List<Integer> i: imageHandler.getFeatures())
            {
                Assert.assertTrue(i.containsAll(imageHandler.computeFeatures().get(c++)));
            }

            ImageHandler imageHandler2 = new ImageHandler("data/testset-19x19/face-png/face00002.png");
            int c2 = 0;
            for (List<Integer> i: imageHandler2.getFeatures())
            {
                Assert.assertTrue(i.containsAll(imageHandler2.computeFeatures().get(c2++)));
            }


            if (Conf.haarExtractor != null)
                Conf.haarExtractor.freeCuda();

        }


    }

}
