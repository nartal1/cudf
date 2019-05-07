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

import java.util.HashSet;
import java.util.Set;

public class CSVOptions {

    public static CSVOptions DEFAULT = new CSVOptions(new Builder());

    // Names of the columns to be returned (other columns are skipped)
    // If empty all columns are returned.
    private final String[] includeColumnNames;

    private CSVOptions(Builder builder) {
        includeColumnNames = builder.includeColumnNames.toArray(
                new String[builder.includeColumnNames.size()]);
    }

    String[] getIncludeColumnNames() {
        return includeColumnNames;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private final Set<String> includeColumnNames = new HashSet<>();

        /**
         * Include a specific column.  Any column not included will not be read.
         * @param name the name of the column.
         */
        public Builder includeColumn(String name) {
            includeColumnNames.add(name);
            return this;
        }

        public CSVOptions build() {
            return new CSVOptions(this);
        }
    }
}