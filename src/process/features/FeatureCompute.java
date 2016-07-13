package process.features;

import java.util.ArrayList;

/**
 * Created by Dubrzr on 13/07/2016.
 */
public class FeatureCompute {
    int[][] integralImage;
    ArrayList<Feature> featuresTypeA = new ArrayList<>();
    ArrayList<Feature> featuresTypeB = new ArrayList<>();
    ArrayList<Feature> featuresTypeC = new ArrayList<>();
    ArrayList<Feature> featuresTypeD = new ArrayList<>();
    ArrayList<Feature> featuresTypeE = new ArrayList<>();

    public FeatureCompute(int[][] integralImage) {
        this.integralImage = integralImage;
        // Compute all
        this.computeTypeA();
        this.computeTypeB();
        this.computeTypeC();
        this.computeTypeD();
        this.computeTypeE();
    }

    public ArrayList<Feature> getAllFeatures() {
        ArrayList<Feature> result = new ArrayList<>();
        result.addAll(this.featuresTypeA);
        result.addAll(this.featuresTypeB);
        result.addAll(this.featuresTypeC);
        result.addAll(this.featuresTypeD);
        result.addAll(this.featuresTypeE);
        return result;
    }

    private void computeTypeA() {
        /**
         * a ------- b ------- c
         * -         -         -
         * -   R1    -    R2   -
         * -         -         -
         * d ------- e ------- f
         * S(R1) = e - (b + d) + a
         * S(R2) = f - (c + e) + b
         */
    }

    private void computeTypeB() {
        /**
         * a ------- b ------- c ------- d
         * -         -         -         -
         * -   R1    -    R2   -    R3   -
         * -         -         -         -
         * e ------- f ------- g ------- h
         * S(R1) = f - (b + e) + a
         * S(R2) = g - (c + f) + b
         * S(R3) = h - (d + g) + c
         */
    }

    private void computeTypeC() {
        /**
         * a ------- b
         * -         -
         * -   R1    -
         * -         -
         * c ------- d
         * -         -
         * -   R2    -
         * -         -
         * e ------- f
         * S(R1) = d - (b + c) + a
         * S(R2) = f - (d + e) + c
         */
    }

    private void computeTypeD() {
        /**
         * a ------- b
         * -         -
         * -   R1    -
         * -         -
         * c ------- d
         * -         -
         * -   R2    -
         * -         -
         * e ------- f
         * -         -
         * -   R3    -
         * -         -
         * g ------- h
         * S(R1) = d - (b + c) + a
         * S(R2) = f - (d + e) + c
         * S(R3) = h - (f + g) + e
         */
    }

    private void computeTypeE() {
        /**
         * a ------- b ------- c
         * -         -         -
         * -   R1    -    R2   -
         * -         -         -
         * d ------- e ------- f
         * -         -         -
         * -   R3    -    R4   -
         * -         -         -
         * g ------- h ------- i
         * S(R1) = e - (b + d) + a
         * S(R2) = f - (c + e) + b
         * S(R3) = h - (e + g) + d
         * S(R4) = i - (f + h) + e
         */
    }
}
