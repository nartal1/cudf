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

/**
 * Options for reading a CSV file
 */
public class CSVOptions extends ColumnFilterOptions {

    public static CSVOptions DEFAULT = new CSVOptions(new Builder());

    private final int headerRow;
    private final byte delim;
    private final byte quote;
    private final byte comment;
    private final String[] nullValues;

    private CSVOptions(Builder builder) {
        super(builder);
        headerRow = builder.headerRow;
        delim = builder.delim;
        quote = builder.quote;
        comment = builder.comment;
        nullValues = builder.nullValues.toArray(
                new String[builder.nullValues.size()]);
    }

    String[] getNullValues() {
        return nullValues;
    }

    int getHeaderRow() {
        return headerRow;
    }

    byte getDelim() {
        return delim;
    }

    byte getQuote() {
        return quote;
    }

    byte getComment() {
        return comment;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder extends ColumnFilterOptions.Builder<Builder> {
        private byte comment = 0;
        private int headerRow = -1;
        private byte delim = ',';
        private byte quote = '"';
        private final Set<String> nullValues = new HashSet<>();

        /**
         * Row of the header data (0 based counting).  Negative is no header.
         */
        public Builder withHeaderAt(int index) {
            headerRow = index;
            return this;
        }

        /**
         * Set the row of the header to 0, the first line.
         */
        public Builder hasHeader() {
            return withHeaderAt(0);
        }

        /**
         * Set the entry deliminator.  Only ASCII chars are currently supported.
         */
        public Builder withDelim(char delim) {
            if (Character.getNumericValue(delim) > 127) {
                throw new IllegalArgumentException("Only ASCII characters are currently supported");
            }
            this.delim = (byte)delim;
            return this;
        }

        /**
         * Set the quote character.  Only ASCII chars are currently supported.
         */
        public Builder withQuote(char quote) {
            if (Character.getNumericValue(quote) > 127) {
                throw new IllegalArgumentException("Only ASCII characters are currently supported");
            }
            this.quote = (byte)quote;
            return this;
        }

        /**
         * Set the character that starts the beginning of a comment line. setting to
         * 0 or '\0' will disable comments. The default is to have no comments.
         */
        public Builder withComment(char comment) {
            if (Character.getNumericValue(quote) > 127) {
                throw new IllegalArgumentException("Only ASCII characters are currently supported");
            }
            this.comment = (byte)comment;
            return this;
        }

        public Builder withoutComments() {
            this.comment = 0;
            return this;
        }

        public Builder withNullValue(String ... nvs) {
            for (String nv : nvs) {
                nullValues.add(nv);
            }
            return this;
        }

        public CSVOptions build() {
            return new CSVOptions(this);
        }
    }
}
