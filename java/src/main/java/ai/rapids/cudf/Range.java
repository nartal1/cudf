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

import java.util.function.Consumer;

/**
 * Helper utility for creating ranges.
 */
public final class Range {
    /**
     * Append a range to the builder. 0 inclusive to end exclusive.
     * @param end last entry exclusive.
     * @return the consumer.
     */
    public static final Consumer<IntColumnVector.Builder> appendInts(int end) {
        return appendInts(0, end, 1);
    }

    /**
     * Append a range to the builder. start inclusive to end exclusive.
     * @param start first entry.
     * @param end last entry exclusive.
     * @return the consumer.
     */
    public static final Consumer<IntColumnVector.Builder> appendInts(int start, int end) {
        return appendInts(start, end, 1);
    }

    /**
     * Append a range to the builder. start inclusive to end exclusive.
     * @param start first entry.
     * @param end last entry exclusive.
     * @param step how must to step by.
     * @return the builder for chaining.
     */
    public static final Consumer<IntColumnVector.Builder> appendInts(int start, int end, int step) {
        assert step > 0;
        assert start <= end;
        return (b) -> {
            for (int i = start; i < end; i += step) {
                b.append(i);
            }
        };
    }
}
