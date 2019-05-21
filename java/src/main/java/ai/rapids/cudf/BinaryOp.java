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
 * Mathematical unary operations.
 */
public enum BinaryOp {
    ADD(0),
    SUB(1),
    MUL(2),
    DIV(3), // divide using common type of lhs and rhs
    TRUE_DIV(4), // divide after promoting to FLOAT64 point
    FLOOR_DIV(5), // divide after promoting to FLOAT64 and flooring the result
    MOD(6),
    POW(7),
    EQUAL(8),
    NOT_EQUAL(9),
    LESS(10),
    GREATER(11),
    LESS_EQUAL(12), // <=
    GREATER_EQUAL(13), // >=
    BITWISE_AND(14),
    BITWISE_OR(15),
    BITWISE_XOR(16);
    //NOT IMPLEMENTED YET COALESCE(17); // x == null ? y : x

    final int nativeId;

    BinaryOp(int nativeId) {
        this.nativeId = nativeId;
    }

    private static final BinaryOp[] OPS = BinaryOp.values();

    static BinaryOp fromNative(int nativeId) {
        for (BinaryOp type: OPS) {
            if (type.nativeId == nativeId) {
                return type;
            }
        }
        throw new IllegalArgumentException("Could not translate " + nativeId + " into a BinaryOp");
    }
}
