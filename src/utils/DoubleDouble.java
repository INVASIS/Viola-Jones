package utils;


import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;

public class DoubleDouble extends BigDecimal {
    public static final MathContext DECIMAL80 = new MathContext(20, RoundingMode.HALF_EVEN);
    public static final DoubleDouble ZERO = new DoubleDouble(0);
    public static final DoubleDouble ONE = new DoubleDouble(1);

    public DoubleDouble(double val) {
        super(val, DECIMAL80);
    }

    public DoubleDouble(BigDecimal val) {
        super(val.toString());
    }

    public DoubleDouble multiplyBy(DoubleDouble val) {
        return new DoubleDouble(this.multiply(val, DECIMAL80));
    }

    public DoubleDouble multiplyBy(double val) {
        return this.multiplyBy(new DoubleDouble(val));
    }

    public DoubleDouble divideBy(DoubleDouble val) {
        // Handles NaN
        if (val.eq(0)) {
            System.err.println("/!\\ divideBy 0 /!\\ ");
            return new DoubleDouble(Double.MAX_VALUE);
        }
        return new DoubleDouble(super.divide(val, DECIMAL80));
    }

    public DoubleDouble divideBy(int val) {
        return this.divideBy(new DoubleDouble(val));
    }

    public DoubleDouble add(DoubleDouble val) {
        return new DoubleDouble(super.add(val, DECIMAL80));
    }

    public DoubleDouble add(double val) {
        return this.add(new DoubleDouble(val));
    }

    public DoubleDouble subtract(DoubleDouble val) {
        return new DoubleDouble(super.subtract(val, DECIMAL80));
    }

    public DoubleDouble subtract(double val) {
        return this.subtract(new DoubleDouble(val));
    }

    public boolean lt(DoubleDouble val) {
        return this.compareTo(val) == -1;
    }

    public boolean lt(double val) {
        return this.lt(new DoubleDouble(val));
    }

    public boolean lte(DoubleDouble val) {
        return this.compareTo(val) <= 0;
    }

    public boolean lte(double val) {
        return this.lte(new DoubleDouble(val));
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
        return new DoubleDouble(super.min(other));
    }

    public DoubleDouble max(DoubleDouble other) {
        return new DoubleDouble(super.max(other));
    }
}
