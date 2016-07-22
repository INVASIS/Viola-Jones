package process.features;

import GUI.ImageHandler;

import java.util.ArrayList;
import java.util.HashMap;

import static process.features.FeatureExtractor.computeFeatures;

public class ImageFeaturesCompute implements Runnable {

    private final ImageHandler image;
    private final HashMap<String, ArrayList<Integer>> result;

    public ImageFeaturesCompute(ImageHandler image, HashMap<String, ArrayList<Integer>> result){
        this.image = image;
        this.result = result;
    }

    @Override
    public void run() {
        result.put(image.getFilePath(), computeFeatures(image));
    }
}
