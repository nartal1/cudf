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
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class Date32ColumnVectorTest {

    @Test
    public void getYear() {

        int length = 5;
        final int val=17897; //Jan 01, 2019
        try (Date32ColumnVector date32ColumnVector = Date32ColumnVector.build(5,
                (b) -> {
                    for (int i=0;i<length;i++){
                        b.append(val - 365 * i);
                    }
                }))
        {
            date32ColumnVector.toDeviceBuffer();
            ShortColumnVector result= date32ColumnVector.year();
            result.toHostBuffer();
            int expected = 2019;
            for(int i =0; i < length; i++) {
                assertEquals(expected-i, result.get(i)); //2019 to 2015
            }
        }
    }

    @Test
    public void getMonth() {
        int length = 5;
        final int val=17897; //Jan 01, 2019
        try (Date32ColumnVector date32ColumnVector = Date32ColumnVector.build(5,
                (b) -> {
                   for (int i=0;i<length;i++){
                       b.append(val - 365 * i);
                   }
                }))
        {
            date32ColumnVector.toDeviceBuffer();
            ShortColumnVector result= date32ColumnVector.month();
            result.toHostBuffer();
            for(int i =0; i < length; i++) {
                assertEquals(1, result.get(i)); //Jan of every year
            }
        }
    }

    @Test
    public void getDay() {
        int length = 5;
        final int val=17897; //Jan 01, 2019
        try (Date32ColumnVector date32ColumnVector = Date32ColumnVector.build(5,
                (b) -> {
                    for (int i=0;i<length;i++){
                        b.append(val + i);
                    }
                }))
        {
            date32ColumnVector.toDeviceBuffer();
            ShortColumnVector result= date32ColumnVector.day();
            result.toHostBuffer();
            for(int i =0; i < length; i++) {
                assertEquals(i+1, result.get(i)); //1 to 5
            }
        }
    }
}
