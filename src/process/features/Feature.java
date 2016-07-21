package process.features;

import GUI.ImageHandler;

import static process.features.FeatureExtractor.*;

public class Feature {
    private final Rectangle r;
    private final int type;
    private int computeValue = -1;
    private final ImageHandler image;

    public Feature(Rectangle r, int type, ImageHandler ih) {
        this.r = r;
        this.type = type;
        this.image = ih;
    }

    public int getValue() {
        if (this.computeValue == -1) {
            if (this.type == typeA)
                this.computeValue = computeTypeA(this.image, this.r);
            else if (this.type == typeB)
                this.computeValue = computeTypeB(this.image, this.r);
            else if (this.type == typeC)
                this.computeValue = computeTypeC(this.image, this.r);
            else if (this.type == typeD)
                this.computeValue = computeTypeD(this.image, this.r);
            else if (this.type == typeE)
                this.computeValue = computeTypeE(this.image, this.r);
        }
        return this.computeValue;
    }

    public Rectangle getRectangle() {
        return r;
    }

    public int getType() {
        return type;
    }
}
