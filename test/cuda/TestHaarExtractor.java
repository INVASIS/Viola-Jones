package cuda;

import GUI.ImageHandler;
import org.junit.Assert;
import org.junit.Test;
import process.Conf;

import java.util.List;

import static process.features.FeatureExtractor.computeImageFeatures;

public class TestHaarExtractor {

    @Test
    public void testComputeFeatures() {

        if (Conf.USE_CUDA) {

            Conf.haarExtractor.setUp(19, 19);
            ImageHandler imageHandler = new ImageHandler("data/trainset/faces/face00001.png");
            int c = 0;
            for (List<Integer> i: imageHandler.getFeatures())
            {
                Assert.assertTrue(i.containsAll(computeImageFeatures(imageHandler.getFilePath(), false).get(c++)));
            }

            ImageHandler imageHandler2 = new ImageHandler("data/testset/faces/cmu_0000.png");
            int c2 = 0;
            for (List<Integer> i: imageHandler2.getFeatures())
            {
                Assert.assertTrue(i.containsAll(computeImageFeatures(imageHandler2.getFilePath(), false).get(c2++)));
            }
        }
    }
}
