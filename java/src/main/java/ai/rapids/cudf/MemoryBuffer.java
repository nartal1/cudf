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
 * Abstract class for representing the Memory Buffer
 */
abstract class MemoryBuffer implements AutoCloseable {
    protected final long address;
    protected final long length;

    /**
     * Public constructor
     * @param address - location in memory
     * @param length - size of this buffer
     */
    public MemoryBuffer(long address, long length) {
        this.address = address;
        this.length = length;
    }

    /**
     * Returns the size of this buffer
     * @return - size
     */
    public long getLength() {
        return length;
    }

    /**
     * Close this buffer and free memory
     */
    public abstract void close();

    @Override
    public String toString() {
        return "MemoryBuffer{" +
                "address=" + address +
                ", rows=" + length +
                '}';
    }
}
