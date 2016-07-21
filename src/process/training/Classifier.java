package process.training;

import jboost.booster.AdaBoost;
import jboost.booster.bag.AdaBoostBinaryBag;
import jboost.booster.prediction.BinaryPrediction;
import jboost.examples.attributes.Label;

import java.awt.image.BufferedImage;
import java.util.*;

import static utils.Utils.listImages;

/**
 * Created by Dubrzr on 20/07/2016.
 */
public class Classifier {
    public void train(String faces_dir, String nonfaces_dir) {
        HashMap<BufferedImage, Boolean> images = new HashMap<>();
        for (BufferedImage i : listImages(faces_dir))
            images.put(i, true);
        for (BufferedImage i : listImages(nonfaces_dir))
            images.put(i, false);

        // Shuffle all images before using them
        ArrayList<BufferedImage> keys = new ArrayList(images.keySet());
        Collections.shuffle(keys);
        for (BufferedImage bi : keys) {
            images.get(bi);
        }
    }

    public static void run() {
        AdaBoost ab = new AdaBoost();
        ab.addExample(0, new Label(1));
        ab.setM_epsilon(0.041666666666666664);
        ab.setM_totalWeight(12.0);

        AdaBoostBinaryBag bag = new AdaBoostBinaryBag(ab);
        bag.setM_w0(7.0);
        bag.setM_w1(5.0);
        BinaryPrediction p = (BinaryPrediction) ab.getPrediction(bag);
        System.out.println("Expecting: " + -0.1550774641519198 + " - Got : " + p.getClassScores()[1]);
        System.out.println("Expecting: " + 0.1550774641519198 + " - Got : " + p.getClassScores()[0]);
    }

    public static void adaboost(Iterable<BufferedImage> faces, Iterable<BufferedImage> nonfaces, int T) {
        // T is the number of rounds the adaboost have to run




        for (int t = 0; t < T; t++) {

        }
    }
}
