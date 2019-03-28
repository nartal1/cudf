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
package ai.rapids.bindings.cudf;

/**
 * Defines the unit of time that an algorithm or structure is storing.
 * <p>
 * There are time types in cudf. Those time types can have different resolutions.
 * The types included are nanosecond, microsecond, millisecond, and second.
 */
public enum CudfTimeUnit {
    NONE(0), // default (undefined)
    SECONDS(1),    // seconds
    MILLISECONDS(2),   // milliseconds
    MICROSECONDS(3),   // microseconds
    NANOSECONDS(4);   // nanoseconds

    private final int value;

    CudfTimeUnit(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }
}
