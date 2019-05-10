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

#include "jni_utils.hpp"
#include "types.hpp"
#include "table.hpp"
#include "copying.hpp"
#include <cstring>

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_CudfTable_createCudfTable(JNIEnv *env,
        jclass classObject,
        jlongArray cudfColumns) {
    JNI_NULL_CHECK(env, cudfColumns, "input columns are null", 0);

    try {
      const cudf::native_jlongArray nCudfColumns(env, cudfColumns);
      int const len = nCudfColumns.size();

      std::unique_ptr<gdf_column *[]> cols(new gdf_column *[len]);
      const jlong* cudfColumnsData = nCudfColumns.data();

      std::memcpy(cols.get(), cudfColumnsData, sizeof(gdf_column*) * len);

      cudf::table* table = new cudf::table(cols.get(), len);
      return reinterpret_cast<jlong>(table);
    } CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_CudfTable_free(JNIEnv *env, jclass classObject, jlong handle) {
    cudf::table* table = reinterpret_cast<cudf::table*>(handle);
    delete table;
}

/**
 * Copy contents of a jbooleanArray into an array of int8_t pointers
 */
cudf::jni_rmm_unique_ptr<int8_t> copyToDevice(JNIEnv * env, const cudf::native_jbooleanArray &nArr) {
  jsize len = nArr.size();
  size_t byteLen = len * sizeof(int8_t);
  const jboolean *tmp = nArr.data();

  std::unique_ptr<int8_t[]> host(new int8_t[byteLen]);

  for (int i = 0; i<len; i++) {
    host[i] = static_cast<int8_t>(nArr[i]);
  }

  auto device = cudf::jniRmmAlloc<int8_t>(env, byteLen);
  cudf::jniCudaCheck(env, cudaMemcpy(device.get(), host.get(), byteLen, cudaMemcpyHostToDevice));
  return device;
}

/**
 * Convert an array of longs into an array of gdf_column pointers.
 */
std::vector<gdf_column *> as_gdf_columns(const cudf::native_jlongArray &nColumnPtrs) {
    jsize numColumns = nColumnPtrs.size();

    std::vector<gdf_column *> columns(numColumns);
    for (int i = 0; i < numColumns; i++) {
      columns[i] = reinterpret_cast<gdf_column*>(nColumnPtrs[i]);
    }
    return columns;
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_CudfTable_gdfOrderBy(JNIEnv *env,
        jclass jClassObject,
        jlong jInputTable,
        jlongArray jSortKeysGdfcolumns,
        jbooleanArray jIsDescending,
        jlong jOutputTable,
        jboolean jAreNullsSmallest) {

    //input validations & verifications
    JNI_NULL_CHECK(env, jInputTable, "input table is null", );
    JNI_NULL_CHECK(env, jSortKeysGdfcolumns, "input table is null", );
    JNI_NULL_CHECK(env, jIsDescending, "sort order array is null", );
    JNI_NULL_CHECK(env, jOutputTable, "output table is null", );

    try {
        const cudf::native_jlongArray nSortKeysGdfcolumns(env, jSortKeysGdfcolumns);
        jsize numColumns = nSortKeysGdfcolumns.size();
        const cudf::native_jbooleanArray nIsDescending(env, jIsDescending);
        jsize numColumnsIsDesc = nIsDescending.size();

        if ( numColumnsIsDesc != numColumns) {
            JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "columns and isDescending lengths don't match", );
        }

        std::vector<gdf_column*> columns = as_gdf_columns(nSortKeysGdfcolumns);
        auto isDescending = copyToDevice(env, nIsDescending);

        cudf::table* outputTable = reinterpret_cast<cudf::table*>(jOutputTable);
        cudf::table* inputTable = reinterpret_cast<cudf::table*>(jInputTable);
        bool areNullsSmallest = static_cast<bool>(jAreNullsSmallest);

        auto col_data = cudf::jniRmmAlloc<int32_t>(env, columns[0]->size * sizeof(int32_t), 0);

        gdf_column intermediateOutput;
        // construct column view
        cudf::jniCudfCheck(env, gdf_column_view(&intermediateOutput, col_data.get(), nullptr, columns[0]->size, gdf_dtype::GDF_INT32));

        gdf_context context{};
        // Most of these are probably ignored, but just to be safe
        context.flag_sorted = false;
        context.flag_method = GDF_SORT;
        context.flag_distinct = 0;
        context.flag_sort_result = 1;
        context.flag_sort_inplace = 0;
        context.flag_groupby_include_nulls = true;
        // There is also a MULTI COLUMN VERSION, that we may want to support in the future.
        context.flag_null_sort_behavior = jAreNullsSmallest ? GDF_NULL_AS_SMALLEST : GDF_NULL_AS_LARGEST;


        cudf::jniCudfCheck(env, gdf_order_by(columns.data(), isDescending.get(), static_cast<size_t>(numColumns), &intermediateOutput, &context));
        
        gather(inputTable, col_data.get(), outputTable);
    } CATCH_STD(env, );
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_CudfTable_gdfReadCSV(JNIEnv* env,
       jclass jClassObject,
       jobjectArray colNames,
       jobjectArray dataTypes,
       jobjectArray filterColNames,
       jstring inputfilepath,
       jlong buffer, jlong bufferLength,
       jint headerRow,
       jbyte delim,
       jbyte quote,
       jbyte comment,
       jobjectArray nullValues) {
    JNI_NULL_CHECK(env, nullValues, "nullValues must be supplied, even if it is empty", NULL);

    bool read_buffer = true;
    if (buffer == 0) {
      JNI_NULL_CHECK(env, inputfilepath, "input file or buffer must be supplied", NULL);
      read_buffer = false;
    } else if (inputfilepath != NULL) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "cannot pass in both a buffer and an inputfilepath", NULL);
    } else if (bufferLength <= 0) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "An empty buffer is not supported", NULL);
    }

    // The number of output columns is determined dynamically when set to 0
    int nColumns = 0;
    try {
      std::unique_ptr<cudf::native_jstringArray> nColNames;
      char const** nameParam = NULL;
      if (colNames != NULL) {
        nColNames.reset(new cudf::native_jstringArray(env, colNames));
        nameParam = nColNames->as_c_array();
        nColumns = nColNames->size();
      }

      std::unique_ptr<cudf::native_jstringArray> nDataTypes;
      char const** dTypes = NULL;
      if (dataTypes != NULL) {
        nDataTypes.reset(new cudf::native_jstringArray(env, dataTypes));
        dTypes = nDataTypes->as_c_array();
        nColumns = nDataTypes->size();
      }

      if (nDataTypes != NULL && nColNames != NULL) {
        int dTypeCount = nDataTypes->size();
        int nameCount = nColNames->size();
        if (dTypeCount != nameCount) {
          JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "dataTypes and colNames should be the same size", NULL);
        }
      }

      std::unique_ptr<cudf::native_jstring> filename;
      if (!read_buffer) {
        filename.reset(new cudf::native_jstring(env, inputfilepath));
        if (strlen(filename->get()) == 0) {
          JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "inputfilepath can't be empty", NULL);
        }
      }

      cudf::native_jstringArray nNullValues(env, nullValues);
      int num_null_values = nNullValues.size();
      char const** c_null_values = NULL;
      if (num_null_values > 0) {
        c_null_values = nNullValues.as_c_array();
      } 

      int numfiltercols = 0;
      char const** filteredNames = NULL;
      std::unique_ptr<cudf::native_jstringArray> nFilterColNames;

      if (filterColNames != NULL) {
        nFilterColNames.reset(new cudf::native_jstringArray(env, filterColNames));
        numfiltercols = nFilterColNames->size();
        filteredNames = nFilterColNames->as_c_array();
      }

      csv_read_arg read_arg{};

      if (read_buffer) {
        read_arg.filepath_or_buffer = reinterpret_cast<const char *>(buffer);
        read_arg.input_data_form = HOST_BUFFER;
        read_arg.buffer_size = bufferLength;
      } else {
        read_arg.filepath_or_buffer = filename->get();

        read_arg.input_data_form = FILE_PATH;
        // don't use buffer, use file path
        read_arg.buffer_size = 0;
      }

      read_arg.windowslinetermination = false;
      read_arg.lineterminator = '\n';
      // delimiter ideally passed in
      read_arg.delimiter = delim;
      read_arg.delim_whitespace = 0;
      read_arg.skipinitialspace = 0;
      read_arg.nrows = -1;
      read_arg.header = headerRow;
      read_arg.num_cols = nColumns;

      read_arg.names = nameParam;
      read_arg.dtype = dTypes;

      // leave blank
      // read_arg.index_col

      // only support picking columns by name
      read_arg.use_cols_int = NULL;
      read_arg.use_cols_int_len = 0;
      read_arg.use_cols_char = filteredNames;

      read_arg.use_cols_char_len = numfiltercols;

      read_arg.skiprows = 0;
      read_arg.skipfooter = 0;
      read_arg.skip_blank_lines = true;

      char const* trueVals[2] = {"True", "TRUE"};
      char const* falseVals[2] = {"False", "FALSE"};
      read_arg.true_values = trueVals;
      read_arg.num_true_values = 2;
      read_arg.false_values = falseVals;
      read_arg.num_false_values = 2;

      read_arg.num_na_values = num_null_values;
      read_arg.na_values = c_null_values;
      read_arg.keep_default_na = false;  ///< Keep the default NA values
      read_arg.na_filter = num_null_values > 0;

      read_arg.prefix = NULL;
      read_arg.mangle_dupe_cols = true;
      read_arg.parse_dates = 1;
      // read_arg.infer_datetime_format = true;
      read_arg.dayfirst = 0;
      read_arg.compression = nullptr;
      // read_arg.thousands
      read_arg.decimal = '.';
      read_arg.quotechar = quote;
      //read_arg.quoting = QUOTE_NONNUMERIC;
      read_arg.quoting = QUOTE_MINIMAL;
      read_arg.doublequote = true;
      // read_arg.escapechar =
      read_arg.comment = comment;
      read_arg.encoding = NULL;
      read_arg.byte_range_offset = 0;
      read_arg.byte_range_size = 0;

      gdf_error gdfStatus = read_csv(&read_arg);
      JNI_GDF_TRY(env, NULL, gdfStatus);

      cudf::native_jlongArray nativeHandles(env,
              reinterpret_cast<jlong*>(read_arg.data),
              read_arg.num_cols_out);
      return nativeHandles.get_jlongArray();
    } CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_CudfTable_gdfReadParquet(JNIEnv* env,
       jclass jClassObject,
       jobjectArray filterColNames,
       jstring inputfilepath,
       jlong buffer, jlong bufferLength) {
    bool read_buffer = true;
    if (buffer == 0) {
      JNI_NULL_CHECK(env, inputfilepath, "input file or buffer must be supplied", NULL);
      read_buffer = false;
    } else if (inputfilepath != NULL) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "cannot pass in both a buffer and an inputfilepath", NULL);
    } else if (bufferLength <= 0) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "An empty buffer is not supported", NULL);
    }

    try {
      std::unique_ptr<cudf::native_jstring> filename;
      if (!read_buffer) {
        filename.reset(new cudf::native_jstring(env, inputfilepath));
        if (strlen(filename->get()) == 0) {
          JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "inputfilepath can't be empty", NULL);
        }
      }

      int numfiltercols = 0;
      char const** filteredNames = NULL;
      std::unique_ptr<cudf::native_jstringArray> nFilterColNames;

      if (filterColNames != NULL) {
        nFilterColNames.reset(new cudf::native_jstringArray(env, filterColNames));
        numfiltercols = nFilterColNames->size();
        filteredNames = nFilterColNames->as_c_array();
      }

      pq_read_arg read_arg{};

      if (read_buffer) {
        read_arg.source = reinterpret_cast<const char *>(buffer);
        read_arg.source_type = HOST_BUFFER;
        read_arg.buffer_size = bufferLength;
      } else {
        read_arg.source = filename->get();
        read_arg.source_type = FILE_PATH;
        // don't use buffer, use file path
        read_arg.buffer_size = 0;
      }

      read_arg.use_cols = filteredNames;
      read_arg.use_cols_len = numfiltercols; 

      read_arg.row_group = -1;
      read_arg.skip_rows = 0;
      read_arg.num_rows = -1;
      read_arg.strings_to_categorical = false;

      gdf_error gdfStatus = read_parquet(&read_arg);
      JNI_GDF_TRY(env, NULL, gdfStatus);

      cudf::native_jlongArray nativeHandles(env,
              reinterpret_cast<jlong*>(read_arg.data),
              read_arg.num_cols_out);
      return nativeHandles.get_jlongArray();
    } CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_CudfTable_gdfLeftJoin(
    JNIEnv *env, jclass clazz, jlong leftTable,
    jintArray leftColJoinIndices, jlong rightTable,
    jintArray rightColJoinIndices) {
  JNI_NULL_CHECK(env, leftTable, "leftTable is null", NULL);
  JNI_NULL_CHECK(env, leftColJoinIndices, "leftColJoinIndices is null", NULL);
  JNI_NULL_CHECK(env, rightTable, "rightTable is null", NULL);
  JNI_NULL_CHECK(env, rightColJoinIndices, "rightColJoinIndices is null", NULL);

  try {
    cudf::table* nLefTable = reinterpret_cast<cudf::table*>(leftTable);
    cudf::table* nRightTable = reinterpret_cast<cudf::table*>(rightTable);
    cudf::native_jintArray leftJoinColsArr(env, leftColJoinIndices);
    cudf::native_jintArray rightJoinColsArr(env, rightColJoinIndices);

    gdf_context context{};
    context.flag_sorted = 0;
    context.flag_method = GDF_HASH;
    context.flag_distinct = 0;
    context.flag_sort_result = 1;
    context.flag_sort_inplace = 0;

    int resultNumCols = nLefTable->num_columns() + nRightTable->num_columns() - leftJoinColsArr.size();

    // gdf_left_join is allocating the memory for the results so
    // allocate the output column structures here when we get it back fill in
    // the the outPtrs
    cudf::native_jlongArray outputHandles(env, resultNumCols);
    std::vector<std::unique_ptr<gdf_column>> output_columns(resultNumCols);
    for (int i = 0; i < resultNumCols; i++) {
        output_columns[i].reset(new gdf_column());
        outputHandles[i] = reinterpret_cast<jlong>(output_columns[i].get());
    }
    JNI_GDF_TRY(env, NULL, gdf_left_join(
        nLefTable->begin(), nLefTable->num_columns(), leftJoinColsArr.data(),
        nRightTable->begin(), nRightTable->num_columns(), rightJoinColsArr.data(),
        leftJoinColsArr.size(), resultNumCols, reinterpret_cast<gdf_column**>(outputHandles.data()), nullptr, nullptr,
        &context));
    for (int i = 0 ; i < resultNumCols ; i++) {
        output_columns[i].release();
    }
    return outputHandles.get_jlongArray();
  }
  CATCH_STD(env, NULL);
}
};