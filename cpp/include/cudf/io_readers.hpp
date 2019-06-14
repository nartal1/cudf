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

#pragma once

#include <string>
#include <vector>
#include <memory>

#include "cudf.h"
#include "table.hpp"

namespace cudf {
 
 /**---------------------------------------------------------------------------*
 * @brief Arguments to the read_json interface.
 *---------------------------------------------------------------------------**/
struct json_reader_args{
  gdf_input_type  source_type = HOST_BUFFER;      ///< Type of the data source.
  std::string     source;                         ///< If source_type is FILE_PATH, contains the filepath. If source_type is HOST_BUFFER, contains the input JSON data.

  std::vector<std::string>  dtype;                ///< Ordered list of data types; pass an empty vector to use data type deduction.
  std::string               compression = "infer";///< Compression type ("none", "infer", "gzip", "zip"); default is "infer".
  bool                      lines = false;        ///< Read the file as a json object per line; default is false.

  json_reader_args() = default;
 
  json_reader_args(json_reader_args const &) = default;

  /**---------------------------------------------------------------------------*
   * @brief json_reader_args constructor that sets the source data members.
   * 
   * @param[in] src_type Enum describing the type of the data source.
   * @param[in] src If src_type is FILE_PATH, contains the filepath.
   * If source_type is HOST_BUFFER, contains the input JSON data.
   *---------------------------------------------------------------------------**/
  json_reader_args(gdf_input_type src_type, std::string const &src) : source_type(src_type), source(src) {}
};

/**---------------------------------------------------------------------------*
 * @brief Class used to parse Json input and convert it into gdf columns.
 *
 *---------------------------------------------------------------------------**/
class JsonReader {
private:
  class Impl;
  std::unique_ptr<Impl> impl_;

public:
  /**---------------------------------------------------------------------------*
   * @brief JsonReader constructor; throws if the arguments are not supported.
   *---------------------------------------------------------------------------**/
  explicit JsonReader(json_reader_args const &args);

  /**---------------------------------------------------------------------------*
   * @brief Parse the input JSON file as specified with the json_reader_args
   * constuctor parameter.
   *
   * @return cudf::table object that contains the array of gdf_columns.
   *---------------------------------------------------------------------------**/
  table read();

  /**---------------------------------------------------------------------------*
   * @brief Parse the input JSON file as specified with the args_ data member.
   *
   * Stores the parsed gdf columns in an internal data member.
   * @param[in] offset ///< Offset of the byte range to read.
   * @param[in] size   ///< Size of the byte range to read. If set to zero,
   * all data after byte_range_offset is read.
   *
   * @return cudf::table object that contains the array of gdf_columns.
   *---------------------------------------------------------------------------**/
  table read_byte_range(size_t offset, size_t size);

  ~JsonReader();
};

/**---------------------------------------------------------------------------*
 * @brief Enumeration of quoting behavior for CSV readers/writers
 *---------------------------------------------------------------------------**/
enum gdf_csv_quote_style{
  QUOTE_MINIMAL,                            ///< Only quote those fields which contain special characters; enable quotation when parsing.
  QUOTE_ALL,                                ///< Quote all fields; enable quotation when parsing.
  QUOTE_NONNUMERIC,                         ///< Quote all non-numeric fields; enable quotation when parsing.
  QUOTE_NONE                                ///< Never quote fields; disable quotation when parsing.
};

/**---------------------------------------------------------------------------*
 * @brief  This struct contains all input parameters to the read_csv function.
 *
 * Parameters are all stored in host memory.
 *
 * Parameters in PANDAS that are unavailable in cudf:
 *   squeeze          - data is always returned as a gdf_column array
 *   engine           - this is the only engine
 *   verbose
 *   keep_date_col    - will not maintain raw data
 *   date_parser      - there is only this parser
 *   float_precision  - there is only one converter that will cover all specified values
 *   dialect          - not used
 *   encoding         - always use UTF-8
 *   escapechar       - always use '\'
 *   parse_dates      - infer date data types and always parse as such
 *   infer_datetime_format - inference not supported

 *---------------------------------------------------------------------------**/
struct csv_reader_args{
  gdf_input_type input_data_form = HOST_BUFFER; ///< Type of source of CSV data
  std::string        filepath_or_buffer;            ///< If input_data_form is FILE_PATH, contains the filepath. If input_data_type is HOST_BUFFER, points to the host memory buffer
  std::string        compression = "infer";         ///< Compression type ("none", "infer", "bz2", "gz", "xz", "zip"); with the default value, "infer", infers the compression from the file extension.

  char          lineterminator = '\n';      ///< Define the line terminator character; Default is '\n'.
  char          delimiter = ',';            ///< Define the field separator; Default is ','.
  char          decimal = '.';              ///< The decimal point character; default is '.'. Should not match the delimiter.
  char          thousands = '\0';           ///< Single character that separates thousands in numeric data; default is '\0'. Should not match the delimiter.
  char          comment = '\0';             ///< The character used to denote start of a comment line. The rest of the line will not be parsed. The default is '\0'.
  bool          dayfirst = false;           ///< Is day the first value in the date format (DD/MM versus MM/DD)? false by default.
  bool          delim_whitespace = false;   ///< Use white space as the delimiter; default is false. This overrides the delimiter argument.
  bool          skipinitialspace = false;   ///< Skip white spaces after the delimiter; default is false.
  bool          skip_blank_lines = true;    ///< Indicates whether to ignore empty lines, or parse and interpret values as NA. Default value is true.
  gdf_size_type header = 0;                 ///< Row of the header data, zero based counting; Default is zero.

  std::vector<std::string> names;           ///< Ordered List of column names; Empty by default.
  std::vector<std::string> dtype;           ///< Ordered List of data types; Empty by default.

  std::vector<int> use_cols_indexes;        ///< Indexes of columns to be processed and returned; Empty by default - process all columns.
  std::vector<std::string> use_cols_names;  ///< Names of columns to be processed and returned; Empty by default - process all columns.

  std::vector<std::string> true_values;     ///< List of values to recognize as boolean True; Empty by default.
  std::vector<std::string> false_values;    ///< List of values to recognize as boolean False; Empty by default.
  std::vector<std::string> na_values;       /**< Array of strings that should be considered as NA. By default the following values are interpreted as NA: 
                                            '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL',
                                            'NaN', 'n/a', 'nan', 'null'. */
  bool          keep_default_na = true;     ///< Keep the default NA values; true by default.
  bool          na_filter = true;           ///< Detect missing values (empty strings and the values in na_values); true by default. Passing false can improve performance.

  std::string   prefix;                     ///< If there is no header or names, prepend this to the column ID as the name; Default value is an empty string.
  bool          mangle_dupe_cols = true;    ///< If true, duplicate columns get a suffix. If false, data will be overwritten if there are columns with duplicate names; true by default.

  char          quotechar = '\"';           ///< Define the character used to denote start and end of a quoted item; default is '\"'.
  gdf_csv_quote_style quoting = QUOTE_MINIMAL; ///< Defines reader's quoting behavior; default is QUOTE_MINIMAL.
  bool          doublequote = true;         ///< Indicates whether to interpret two consecutive quotechar inside a field as a single quotechar; true by default.

  csv_reader_args() = default;
};

/**---------------------------------------------------------------------------*
 * @brief Class used to parse Json input and convert it into gdf columns.
 *
 *---------------------------------------------------------------------------**/
class CsvReader {
private:
  class Impl;
  std::unique_ptr<Impl> impl_;

public:
  /**---------------------------------------------------------------------------*
   * @brief CsvReader constructor; throws if the arguments are not supported.
   *---------------------------------------------------------------------------**/
  explicit CsvReader(csv_reader_args const &args);

  /**---------------------------------------------------------------------------*
   * @brief Parse the input CSV file as specified with the csv_reader_args
   * constuctor parameter.
   *
   * @return cudf::table object that contains the array of gdf_columns.
   *---------------------------------------------------------------------------**/
  table read();

  /**---------------------------------------------------------------------------*
   * @brief Parse the specified byte range of the input CSV file.
   *
   * Reads the row that starts before or at the end of the range, even if it ends
   * after the end of the range.
   *
   * @param[in] offset Offset of the byte range to read.
   * @param[in] size Size of the byte range to read. Set to zero to read to
   * the end of the file.
   *
   * @return cudf::table object that contains the array of gdf_columns
   *---------------------------------------------------------------------------**/
  table read_byte_range(size_t offset, size_t size);

  /**---------------------------------------------------------------------------*
   * @brief Parse the specified rows of the input CSV file.
   * 
   * Set num_skip_footer to zero when using num_rows parameter.
   *
   * @param[in] num_skip_header Number of rows at the start of the files to skip.
   * @param[in] num_skip_footer Number of rows at the bottom of the file to skip.
   * @param[in] num_rows Number of rows to read. Value of -1 indicates all rows.
   * 
   * @return cudf::table object that contains the array of gdf_columns
   *---------------------------------------------------------------------------**/
  table read_rows(gdf_size_type num_skip_header, gdf_size_type num_skip_footer, gdf_size_type num_rows = -1);

  ~CsvReader();
};

} // namespace cudf
