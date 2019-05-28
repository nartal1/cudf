/*
 *
 *  Copyright (c) 2019, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package ai.rapids.cudf;

import java.util.Objects;

/**
 * A single scalar value.
 */
public final class Scalar implements BinaryOperable {
    /**
     * Generic NULL value.
     */
    public static final Scalar NULL = new Scalar(DType.INT8, TimeUnit.NONE);

    /*
     * In the native code all of the value are stored in a union with separate entries for each
     * DType.  Java has no equivalent to a union, so as a space saving compromise we store all
     * possible integer types (INT8 - INT64, DATE, TIMESTAMP, etc) in intTypeStorage.
     * Because conversion between a float and a double is not as cheap as it is for integers, we
     * split out float and double into floatTypeStorage and doubleTypeStorage.
     */
    final long intTypeStorage;
    final float floatTypeStorage;
    final double doubleTypeStorage;
    final DType type;
    final boolean isValid;
    // TimeUnit is not currently used by scalar values.  There are no operations that need it
    // When this changes we can support it.
    final TimeUnit timeUnit;

    private Scalar(long value, DType type, TimeUnit unit) {
        intTypeStorage = value;
        floatTypeStorage = 0;
        doubleTypeStorage = 0;
        this.type = type;
        isValid = true;
        timeUnit = unit;
    }

    private Scalar(float value, DType type, TimeUnit unit) {
        intTypeStorage = 0;
        floatTypeStorage = value;
        doubleTypeStorage = 0;
        this.type = type;
        isValid = true;
        timeUnit = unit;
    }

    private Scalar(double value, DType type, TimeUnit unit) {
        intTypeStorage = 0;
        floatTypeStorage = 0;
        doubleTypeStorage = value;
        this.type = type;
        isValid = true;
        timeUnit = unit;
    }

    private Scalar(DType type, TimeUnit unit) {
        intTypeStorage = 0;
        floatTypeStorage = 0;
        doubleTypeStorage = 0;
        this.type = type;
        isValid = false;
        timeUnit = unit;
    }

    // These are invoked by native code to construct scalars.
    static Scalar fromNull(int dtype) {
        return new Scalar(DType.fromNative(dtype), TimeUnit.NONE);
    }

    static Scalar timestampFromNull(int nativeTimeUnit) {
        return timestampFromNull(TimeUnit.fromNative(nativeTimeUnit));
    }

    static Scalar timestampFromLong(long value, int nativeTimeUnit) {
        return timestampFromLong(value, TimeUnit.fromNative(nativeTimeUnit));
    }

    // These Scalar factory methods are called from native code.
    // If a new scalar type is supported then CudfJni also needs to be updated.

    public static Scalar fromNull(DType dtype) {
        return new Scalar(dtype, TimeUnit.NONE);
    }

    public static Scalar timestampFromNull(TimeUnit timeUnit) {
        return new Scalar(DType.TIMESTAMP, timeUnit);
    }

    public static Scalar fromBool(boolean value) {
        return new Scalar(value ? 1 : 0, DType.BOOL8, TimeUnit.NONE);
    }

    public static Scalar fromByte(byte value) {
        return new Scalar(value, DType.INT8, TimeUnit.NONE);
    }

    public static Scalar fromShort(short value) {
        return new Scalar(value, DType.INT16, TimeUnit.NONE);
    }

    public static Scalar fromInt(int value) {
        return new Scalar(value, DType.INT32, TimeUnit.NONE);
    }

    public static Scalar dateFromInt(int value) {
        return new Scalar(value, DType.DATE32, TimeUnit.NONE);
    }

    public static Scalar fromLong(long value) {
        return new Scalar(value, DType.INT64, TimeUnit.NONE);
    }

    public static Scalar dateFromLong(long value) {
        return new Scalar(value, DType.DATE64, TimeUnit.NONE);
    }

    public static Scalar timestampFromLong(long value) {
        return new Scalar(value, DType.TIMESTAMP, TimeUnit.MILLISECONDS);
    }

    public static Scalar timestampFromLong(long value, TimeUnit unit) {
        if (unit == TimeUnit.NONE) {
            unit = TimeUnit.MILLISECONDS;
        }
        return new Scalar(value, DType.TIMESTAMP, unit);
    }

    public static Scalar fromFloat(float value) {
        return new Scalar(value, DType.FLOAT32, TimeUnit.NONE);
    }

    public static Scalar fromDouble(double value) {
        return new Scalar(value, DType.FLOAT64, TimeUnit.NONE);
    }

    public boolean isValid() {
        return isValid;
    }

    @Override
    public DType getType() {
        return type;
    }

    /**
     * Returns the boolean stored in this scalar. Only valid if the type is
     * {@link DType#BOOL8}.
     */
    public boolean getBoolean() {
        assert type == DType.BOOL8;
        return intTypeStorage != 0;
    }

    /**
     * Returns the byte stored in this scalar. Only valid if the type is
     * {@link DType#BOOL8}.
     */
    public byte getByte() {
        assert type == DType.INT8;
        return (byte) intTypeStorage;
    }

    /**
     * Returns the short stored in this scalar. Only valid if the type is
     * {@link DType#INT16}.
     */
    public short getShort() {
        assert type == DType.INT16;
        return (short) intTypeStorage;
    }

    /**
     * Returns the int stored in this scalar. Only valid if the type is
     * {@link DType#INT32} or {@link DType#DATE32}.
     */
    public int getInt() {
        assert type == DType.INT32 || type == DType.DATE32;
        return (int) intTypeStorage;
    }

    /**
     * Returns the long stored in this scalar. Only valid if the type is
     * {@link DType#INT64}, {@link DType#DATE64}, or {@link DType#TIMESTAMP}.
     */
    public long getLong() {
        assert type == DType.INT64 || type == DType.DATE64 || type == DType.TIMESTAMP;
        return intTypeStorage;
    }

    /**
     * Returns the float stored in this scalar. Only valid if the type is
     * {@link DType#FLOAT32}.
     */
    public float getFloat() {
        assert type == DType.FLOAT32;
        return floatTypeStorage;
    }

    /**
     * Returns the double stored in this scalar. Only valid if the type is
     * {@link DType#FLOAT64}.
     */
    public double getDouble() {
        assert type == DType.FLOAT64;
        return doubleTypeStorage;
    }

    /**
     * Returns the time units associated with this scalar. Only valid if the
     * type is {@link DType#TIMESTAMP}.
     */
    public TimeUnit getTimeUnit() {
        assert type == DType.TIMESTAMP;
        return timeUnit;
    }

    @Override
    public ColumnVector binaryOp(BinaryOp op, BinaryOperable rhs, DType outType) {
        if (rhs instanceof ColumnVector) {
            ColumnVector cvRhs = (ColumnVector) rhs;
            return new ColumnVector(Cudf.gdfBinaryOp(this, cvRhs, op, outType));
        } else {
            throw new IllegalArgumentException(rhs.getClass() + " is not supported as a binary op with Scalar");
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Scalar scalar = (Scalar) o;
        return intTypeStorage == scalar.intTypeStorage &&
            Float.compare(scalar.floatTypeStorage, floatTypeStorage) == 0 &&
            Double.compare(scalar.doubleTypeStorage, doubleTypeStorage) == 0 &&
            isValid == scalar.isValid &&
            type == scalar.type &&
            timeUnit == scalar.timeUnit;
    }

    @Override
    public int hashCode() {
        return Objects.hash(intTypeStorage, floatTypeStorage, doubleTypeStorage, type, isValid, timeUnit);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Scalar{type=");
        sb.append(type);
        sb.append(" value=");

        switch (type) {
        case BOOL8:
            sb.append(getBoolean());
            break;
        case INT8:
            sb.append(getByte());
            break;
        case INT16:
            sb.append(getShort());
            break;
        case INT32:
        case DATE32:
            sb.append(getInt());
            break;
        case INT64:
        case DATE64:
            sb.append(getLong());
            break;
        case FLOAT32:
            sb.append(getFloat());
            break;
        case FLOAT64:
            sb.append(getDouble());
            break;
        case TIMESTAMP:
            sb.append(getLong());
            sb.append(" unit=");
            sb.append(getTimeUnit());
            break;
        default:
            throw new IllegalArgumentException("Unknown scalar type: " + type);
        }

        sb.append("}");
        return sb.toString();
    }
}
