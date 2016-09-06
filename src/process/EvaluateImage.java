package process;

import utils.Serializer;
import utils.Utils;

import java.util.ArrayList;

import static process.features.FeatureExtractor.computeImageFeatures;
import static utils.Serializer.readArrayFromDisk;

public class EvaluateImage {

    private int countTestPos;
    private int countTestNeg;
    private int testN;

    private String directory;

    private ArrayList<StumpRule> rules;
    private ArrayList<Integer> layerCommitteeSize;
    private ArrayList<Float> tweaks;
    private int layerCount;

    private ArrayList<StumpRule>[] cascade;

    public EvaluateImage(int countTestPos, int countTestNeg, String directory) {
        this.countTestPos = countTestPos;
        this.countTestNeg = countTestNeg;
        this.testN = countTestNeg + countTestPos;
        this.directory = directory;

        init();
    }

    private void init() {
        this.rules = Serializer.readRule(Conf.TRAIN_FEATURES);

        this.layerCommitteeSize = new ArrayList<>();
        this.tweaks = new ArrayList<>();
        this.layerCount = Serializer.readLayerMemory(Conf.TRAIN_FEATURES, this.layerCommitteeSize, this.tweaks);

        cascade = new ArrayList[this.layerCount];

        int committeeStart = 0;
        for (int i = 0; i < this.layerCount; i++) {
            cascade[i] = new ArrayList<>();
            for (int committeeIndex = committeeStart; committeeIndex < this.layerCommitteeSize.get(i) + committeeStart; committeeIndex++) {
                cascade[i].add(rules.get(committeeIndex));
            }
            committeeStart += this.layerCommitteeSize.get(i);
        }
    }

    public boolean guess(String fileName) {

        // Handle the case this is not a haar file
        ArrayList<Integer> haar;
        if (fileName.endsWith(Conf.IMAGES_EXTENSION) || !Utils.fileExists(fileName)) {
            haar = computeImageFeatures(fileName, false);
        } else {
            haar = readArrayFromDisk(fileName);
        }

        return Classifier.isFace(this.cascade, this.tweaks, haar, this.layerCount);

    }




}
