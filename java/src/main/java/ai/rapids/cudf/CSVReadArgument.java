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

import java.util.ArrayList;
import java.util.List;

public class CSVReadArgument {

    private String filepath;                            // csv file path
    private final String[] filterColumnNames;            // Names of columns to be returned. CSV reader will only process those columns, another read is needed to get full data
    private final String[] columnNames;                    // Ordered List of column columnNames
    private final String[] dTypes;                    // Ordered List of data types

    private CSVReadArgument(String filepath, List<Builder.ColumnType> columnTypes, int filterLength) {
        assert !filepath.isEmpty() : "Filepath is empty";
        this.filepath = filepath;
        this.columnNames = new String[columnTypes.size()];
        this.dTypes = new String[columnTypes.size()];
        this.filterColumnNames = new String[filterLength];
        for (int i = 0, j = 0 ; i < columnTypes.size() ; i++) {
            columnNames[i] = "id" + i;
            Builder.ColumnType columnType = columnTypes.get(i);
            dTypes[i] = columnType.type.simpleName;
            if (columnType.isFilter) {
                assert j < filterLength : "over ran the filter length array";
                filterColumnNames[j++] = columnNames[i];
            }
        }
    }

    String getFilepath() {
        return filepath;
    }

    String[] getFilterColumnNames() {
        return filterColumnNames;
    }

    String[] getColumnNames() {
        return columnNames;
    }

    String[] getDTypes() {
        return dTypes;
    }

    /**
     * Factory method to create builder with filepath
     *
     * @param file File path
     * @return Builder based on filepath
     */
    public static Builder createBuilderWithFilePath(String file) {
        Builder b = new Builder();
        b.filepath = file;
        return b;
    }

    public static class Builder {

        /**
         * Input arguments being used in the POC
         */
        private String filepath;
        private int filterColumnsLength;
        private List<ColumnType> dTypes;

        Builder() {
            dTypes = new ArrayList<>();
        }

        public Builder addColumn(DType type) {
            dTypes.add(new ColumnType(type, true));
            filterColumnsLength++;
            return this;
        }

        public Builder skipColumn() {
            dTypes.add(new ColumnType(/*type is irrelevant*/ DType.INT32, false));
            return this;
        }

        public CSVReadArgument build() {
            return new CSVReadArgument(filepath, dTypes, filterColumnsLength);
        }

        private static class ColumnType {
            DType type;
            boolean isFilter;
            ColumnType(DType type, boolean isFilter) {
                this.type = type;
                this.isFilter = isFilter;
            }
        }
    }
}