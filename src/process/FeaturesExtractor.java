package process;

import process.features.Feature;

import java.util.ArrayList;

public class FeaturesExtractor {

    public static int[][] summedAreaTable(int[][] image, int width, int height) {
        int[][] result = new int[width][height];

        // Array copy
        for (int x = 0; x < width; x++)
            System.arraycopy(image[x], 0, result[x], 0, height);


        // Top border
        for (int x = 1; x < width; x++)
            result[x][0] = result[x][0] + result[x - 1][0];

        // Left border
        for (int y = 1; y < height; y++)
            result[0][y]  = result[0][y] + result[0][y - 1];

        // Remaining pixels
        for (int x = 1; x < width; x++)
            for (int y = 1; y < height; y++)
                result[x][y] = result[x][y] + result[x - 1][y] + result[x][y - 1] - result[x - 1][y - 1];

        return result;
    }

    // Warning : this does not compute the mean of the image, just the sum of pixels
    // To have the mean you must divide by the number of pixels in your rectangle
    public static int rectangleSum(int[][] summedAeraTable, int x, int y, int width, int height) {

        int A = x > 0 && y > 0 ? summedAeraTable[x - 1][y - 1] : 0;
        int B = x + width > 0 && y > 0 ? summedAeraTable[x + width - 1][y - 1] : 0;
        int C = x > 0 && y + height > 0 ? summedAeraTable[x - 1][y + height - 1] : 0;
        int D = x + width > 0 && y + height > 0 ? summedAeraTable[x + width - 1][y + height - 1] : 0;

        return A + D - B - C;
    }

    public static ArrayList<Feature> computeHaarFeatures(int[][] grayImage) {
        ArrayList<Feature> features = new ArrayList<>();
        return features;
    }
}
