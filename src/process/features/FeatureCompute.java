package process.features;

import java.util.ArrayList;

/**
 * Created by Dubrzr on 13/07/2016.
 */
public class FeatureCompute {
    int[][] frame;

    public FeatureCompute(int[][] frame) {
        this.frame = frame;
    }

    public ArrayList<Feature> compute() {
        ArrayList<Feature> features = this.computeTypeA();
        features.addAll(this.computeTypeB());
        features.addAll(this.computeTypeC());
        features.addAll(this.computeTypeD());
        features.addAll(this.computeTypeE());

        return features;
    }

    private ArrayList<Feature> computeTypeA() {
        ArrayList<Feature> features = new ArrayList<Feature>();
        return features;
    }

    private ArrayList<Feature> computeTypeB() {
        ArrayList<Feature> features = new ArrayList<Feature>();
        return features;
    }

    private ArrayList<Feature> computeTypeC() {
        ArrayList<Feature> features = new ArrayList<Feature>();
        return features;
    }

    private ArrayList<Feature> computeTypeD() {
        ArrayList<Feature> features = new ArrayList<Feature>();
        return features;
    }

    private ArrayList<Feature> computeTypeE() {
        ArrayList<Feature> features = new ArrayList<Feature>();
        return features;
    }
}
