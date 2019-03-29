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

    /* arith */

    static void gdfAddGeneric(CudfColumn lhs, CudfColumn rhs, CudfColumn output) {
        gdfAddGeneric(lhs.nativeHandle, rhs.nativeHandle, output.nativeHandle);
    }

    private static native void gdfAddGeneric(long lhs, long rhs, long output) throws CudfException;


    static void gdfAddI32(CudfColumn lhs, CudfColumn rhs, CudfColumn output) {
        gdfAddI32(lhs.nativeHandle, rhs.nativeHandle, output.nativeHandle);
    }

    private static native void gdfAddI32(long lhs, long rhs, long output) throws CudfException;


    static void gdfAddI64(CudfColumn lhs, CudfColumn rhs, CudfColumn output) {
        gdfAddI64(lhs.nativeHandle, rhs.nativeHandle, output.nativeHandle);
    }

    private static native void gdfAddI64(long lhs, long rhs, long output) throws CudfException;


    static void gdfAddF32(CudfColumn lhs, CudfColumn rhs, CudfColumn output) {
        gdfAddF32(lhs.nativeHandle, rhs.nativeHandle, output.nativeHandle);
    }

    private static native void gdfAddF32(long lhs, long rhs, long output) throws CudfException;


    static void gdfAddF64(CudfColumn lhs, CudfColumn rhs, CudfColumn output) {
        gdfAddF64(lhs.nativeHandle, rhs.nativeHandle, output.nativeHandle);
    }

    private static native void gdfAddF64(long lhs, long rhs, long output) throws CudfException;
}
