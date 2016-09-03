package utils;


import java.math.BigDecimal;
import java.math.MathContext;

public class DoubleDouble extends BigDecimal {
    public static final DoubleDouble ZERO = new DoubleDouble(0);
    public static final DoubleDouble ONE = new DoubleDouble(1);

    public DoubleDouble(double val) {
        super(val, MathContext.DECIMAL128);
    }

    public DoubleDouble multiplyBy(DoubleDouble val) {
        return (DoubleDouble) this.multiply(val, MathContext.DECIMAL128);
    }

    public DoubleDouble multiplyBy(double val) {
        return this.multiplyBy(new DoubleDouble(val));
    }

    public DoubleDouble divideBy(DoubleDouble val) {
        // Handles NaN
        if (val.eq(0))
            return new DoubleDouble(Double.MAX_VALUE);
        return (DoubleDouble) super.divide(val, MathContext.DECIMAL128);
    }

    public DoubleDouble divideBy(int val) {
        return this.divideBy(new DoubleDouble(val));
    }

    public DoubleDouble add(DoubleDouble val) {
        return (DoubleDouble) super.add(val, MathContext.DECIMAL128);
    }

    public DoubleDouble add(double val) {
        return this.add(new DoubleDouble(val));
    }

    public DoubleDouble subtract(DoubleDouble val) {
        return (DoubleDouble) super.subtract(val, MathContext.DECIMAL128);
    }

    public DoubleDouble subtract(double val) {
        return this.subtract(new DoubleDouble(val));
    }

    public boolean lt(DoubleDouble val) {
        return this.compareTo(val) == -1;
    }

    public boolean lte(DoubleDouble val) {
        return this.compareTo(val) <= 0;
    }

    public boolean eq(DoubleDouble val) {
        return this.compareTo(val) == 0;
    }

    public boolean eq(double val) {
        return this.eq(new DoubleDouble(val));
    }

    public boolean gte(DoubleDouble val) {
        return this.compareTo(val) >= 0;
    }

    public boolean gte(double val) {
        return this.gte(new DoubleDouble(val));
    }

    public boolean gt(DoubleDouble val) {
        return this.compareTo(val) == 1;
    }

    public DoubleDouble min(DoubleDouble other) {
        return (DoubleDouble) super.min(other);
    }

    public DoubleDouble max(DoubleDouble other) {
        return (DoubleDouble) super.max(other);
    }
}
