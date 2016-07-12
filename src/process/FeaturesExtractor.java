package process;

public class FeaturesExtractor {

    public static int[][] summedAreaTable(int[][] image, int width, int height) {
        int[][] result = new int[width][height];

        for (int x = 0; x < width; x++) {
            System.arraycopy(image[x], 0, result[x], 0, height);
        }

        for (int x = 1; x < width; x++) {

            int i = result[x][0] + result[x - 1][0];
            result[x][0] = i;
        }

        for (int y = 1; y < height; y++) {

            int i = result[0][y] + result[0][y - 1];
            result[0][y]  = i;
        }

        for (int x = 1; x < width; x++) {
            for (int y = 1; y < height; y++) {
                int i = result[x][y] + result[x - 1][y] + result[x][y - 1] - result[x - 1][y - 1];
                result[x][y] = i;
            }
        }

        return result;
    }

    public static int rectangleMean(int[][] summedAeraTable, int x, int y, int width, int height) {

        int A = x > 0 && y > 0 ? summedAeraTable[x - 1][y - 1] : 0;
        int B = x + width > 0 && y > 0 ? summedAeraTable[x + width - 1][y - 1] : 0;
        int C = x > 0 && y + height > 0 ? summedAeraTable[x - 1][y + height - 1] : 0;
        int D = x + width > 0 && y + height > 0 ? summedAeraTable[x + width - 1][y + height - 1] : 0;

        return A + D - B - C;
    }

}
