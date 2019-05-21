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

/**
 * A single scalar value.
 */
public final class Scalar implements BinaryOperable {
    static {
        NativeDepsLoader.loadNativeDeps();
    }

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

    @Override
    public DType getType() {
        return type;
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
}
