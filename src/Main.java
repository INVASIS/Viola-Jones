import GUI.ImageHandler;
import process.Conf;
import utils.Utils;

import java.io.File;
import java.util.ArrayList;
import java.util.List;


/**
 * this is a main class
 */
public class Main {
    public static void main(String[] args) {

        // TODO : TO CONSTANTS
        float overallTargetDetectionRate = 0.80f;
        float overallTargetFalsePositiveRate = 0.000001f;
        float targetDetectionRate = 0.995f;
        float targetFalsePositiveRate = 0.5f;

        if (Conf.USE_CUDA) {
            Conf.haarExtractor.setUp(19, 19);
        }

        // Compute the list of files to give them an index
        File directoryPos = new File("data/testset-19x19/face-png");
        File directoryNeg = new File("data/testset-19x19/non-face-png");

        String[] fileListPos = directoryPos.list();
        String[] fileListNeg = directoryNeg.list();
        ArrayList<String> fileList = new ArrayList<>();
        for (String aFileListNeg : fileListNeg) {
            if (!aFileListNeg.endsWith(Conf.FEATURE_EXTENSION)) {
                fileList.add("non-face-png/" + aFileListNeg);
            }
        }
        for (String fileListPo : fileListPos) {
            if (!fileListPo.endsWith(Conf.FEATURE_EXTENSION)) {
                fileList.add("face-png/" + fileListPo);
            }
        }
        String[] filelistIndex = new String[fileList.size()];
        fileList.toArray(filelistIndex);
        // Compute the list of files to give them an index

        //ImageHandler imageHandler = new ImageHandler("data/face.jpg");
        ImageHandler imageHandler = new ImageHandler("data/testset-19x19/face-png/face00001.png");

//        Display.drawImage(imageHandler.getBufferedImage());
//        Display.drawImage(imageHandler.getGrayBufferedImage());

//        AnyFilter filter = new AnyFilter(imageHandler.getWidth(), imageHandler.getHeight(), imageHandler.getGrayImage());
//        filter.compute();
//
        //HaarExtractor haarExtractor = new HaarExtractor(imageHandler.getWidth(), imageHandler.getHeight());
        //haarExtractor.updateImage(imageHandler.getIntegralImage());
        //haarExtractor.compute();

//        FeatureExtractor fc = new FeatureExtractor(imageHandler);
//        ArrayList<Feature> features = fc.getAllFeatures();

        // Verif Should be all true
        int c = 0;
        for (List<Integer> i : imageHandler.getFeatures())
            System.out.println(i.containsAll(imageHandler.computeFeatures().get(c++)));

        // Classifier.train("data/testset-19x19", "data/testset-19x19", 19, 19);

        Utils.computeHaar(new File("data/testset-19x19/face-png"));
        Utils.computeHaar(new File("data/testset-19x19/non-face-png"));

        if (Conf.haarExtractor != null)
            Conf.haarExtractor.freeCuda();

    }
}
