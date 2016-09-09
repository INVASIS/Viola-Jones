package process.features;

public class Face extends Rectangle {

    double confidence;

    public Face(Rectangle rectangle, double confidence) {
        super(rectangle.getX(), rectangle.getY(), rectangle.getWidth(), rectangle.getHeight());

        this.confidence = confidence;
    }

    public double getConfidence() {
        return confidence;
    }
}
