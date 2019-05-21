/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.rapids.cudf;

/**
 * This is the binding class for cudf lib.
 */
class Cudf {
    static {
        NativeDepsLoader.loadNativeDeps();
    }

    /* arith */

    static long gdfUnaryMath(ColumnVector input, UnaryOp op, DType outputType) {
        return gdfUnaryMath(input.getNativeCudfColumnAddress(), op.nativeId, outputType.nativeId);
    }

    private static native long gdfUnaryMath(long input, int op, int dtype);

    static long gdfBinaryOp(ColumnVector lhs, ColumnVector rhs, BinaryOp op, DType outputType) {
        return gdfBinaryOpVV(lhs.getNativeCudfColumnAddress(), rhs.getNativeCudfColumnAddress(),
                op.nativeId, outputType.nativeId);
    }

    private static native long gdfBinaryOpVV(long lhs, long rhs, int op, int dtype);

    static long gdfBinaryOp(Scalar lhs, ColumnVector rhs, BinaryOp op, DType outputType) {
        return gdfBinaryOpSV(lhs.intTypeStorage, lhs.floatTypeStorage, lhs.doubleTypeStorage, lhs.isValid, lhs.type.nativeId,
                rhs.getNativeCudfColumnAddress(), op.nativeId, outputType.nativeId);
    }

    private static native long gdfBinaryOpSV(long lhsIntValues, float lhsFValue, double lhsDValue, boolean lhsIsValid, int lhsDtype,
                                             long rhs,
                                             int op, int dtype);

    static long gdfBinaryOp(ColumnVector lhs, Scalar rhs, BinaryOp op, DType outputType) {
        return gdfBinaryOpVS(lhs.getNativeCudfColumnAddress(),
                rhs.intTypeStorage, rhs.floatTypeStorage, rhs.doubleTypeStorage, rhs.isValid, rhs.type.nativeId,
                op.nativeId, outputType.nativeId);
    }

    private static native long gdfBinaryOpVS(long lhs,
                                             long rhsIntValues, float rhsFValue, double rhsDValue, boolean rhsIsValid, int rhsDtype,
                                             int op, int dtype);

    /* datetime extract*/

    static long gdfExtractDatetimeYear(ColumnVector input) {
        return gdfExtractDatetimeYear(input.getNativeCudfColumnAddress());
    }

    private static native long gdfExtractDatetimeYear(long input) throws CudfException;

    static long gdfExtractDatetimeMonth(ColumnVector input) {
        return gdfExtractDatetimeMonth(input.getNativeCudfColumnAddress());
    }

    private static native long gdfExtractDatetimeMonth(long input) throws CudfException;

    static long gdfExtractDatetimeDay(ColumnVector input) {
        return gdfExtractDatetimeDay(input.getNativeCudfColumnAddress());
    }

    private static native long gdfExtractDatetimeDay(long input) throws CudfException;

    static long gdfExtractDatetimeHour(ColumnVector input) {
        return gdfExtractDatetimeHour(input.getNativeCudfColumnAddress());
    }

    private static native long gdfExtractDatetimeHour(long input) throws CudfException;

    static long gdfExtractDatetimeMinute(ColumnVector input) {
        return gdfExtractDatetimeMinute(input.getNativeCudfColumnAddress());
    }

    private static native long gdfExtractDatetimeMinute(long input) throws CudfException;

    static long gdfExtractDatetimeSecond(ColumnVector input) {
        return gdfExtractDatetimeSecond(input.getNativeCudfColumnAddress());
    }

    private static native long gdfExtractDatetimeSecond(long input) throws CudfException;
}
