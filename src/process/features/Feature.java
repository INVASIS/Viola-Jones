package process.features;

/**
 * Created by Dubrzr on 13/07/2016.
 */
public class Feature {
    private final Rectangle r;
    private final int type;
    private final int computeValue;

    public Feature(Rectangle r, int type, int computeValue) {
        this.r = r;
        this.type = type;
        this.computeValue = computeValue;
    }

    public int getComputeValue() {
        return computeValue;
    }

    public Rectangle getRectangle() {
        return r;
    }

    public int getType() {
        return type;
    }
}
