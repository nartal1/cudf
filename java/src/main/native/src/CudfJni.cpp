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
#include "reduction.hpp"

using unique_nvcat_ptr = std::unique_ptr<NVCategory, decltype(&NVCategory::destroy)>;
using unique_nvstr_ptr = std::unique_ptr<NVStrings, decltype(&NVStrings::destroy)>;

namespace cudf {
namespace jni {

static const jint MINIMUM_JNI_VERSION = JNI_VERSION_1_6;

static jclass Scalar_jclass;
static jmethodID Scalar_fromNull;
static jmethodID Scalar_timestampFromNull;
static jmethodID Scalar_fromBool;
static jmethodID Scalar_fromByte;
static jmethodID Scalar_fromShort;
static jmethodID Scalar_fromInt;
static jmethodID Scalar_dateFromInt;
static jmethodID Scalar_fromLong;
static jmethodID Scalar_dateFromLong;
static jmethodID Scalar_timestampFromLong;
static jmethodID Scalar_fromFloat;
static jmethodID Scalar_fromDouble;

#define SCALAR_CLASS "ai/rapids/cudf/Scalar"
#define SCALAR_FACTORY_SIG(param_sig) "(" param_sig ")L" SCALAR_CLASS ";"

// Cache useful method IDs of the Scalar class along with a global reference
// to the class. This avoids redundant, dynamic class and method lookups later.
// Returns true if the class and method IDs were successfully cached or false
// if an error occurred.
static bool cache_scalar_jni(JNIEnv *env) {
  jclass cls = env->FindClass(SCALAR_CLASS);
  if (cls == nullptr) {
    return false;
  }

  Scalar_fromNull = env->GetStaticMethodID(cls, "fromNull", SCALAR_FACTORY_SIG("I"));
  if (Scalar_fromNull == nullptr) {
    return false;
  }
  Scalar_timestampFromNull =
      env->GetStaticMethodID(cls, "timestampFromNull", SCALAR_FACTORY_SIG("I"));
  if (Scalar_timestampFromNull == nullptr) {
    return false;
  }
  Scalar_fromBool = env->GetStaticMethodID(cls, "fromBool", SCALAR_FACTORY_SIG("Z"));
  if (Scalar_fromBool == nullptr) {
    return false;
  }
  Scalar_fromByte = env->GetStaticMethodID(cls, "fromByte", SCALAR_FACTORY_SIG("B"));
  if (Scalar_fromByte == nullptr) {
    return false;
  }
  Scalar_fromShort = env->GetStaticMethodID(cls, "fromShort", SCALAR_FACTORY_SIG("S"));
  if (Scalar_fromShort == nullptr) {
    return false;
  }
  Scalar_fromInt = env->GetStaticMethodID(cls, "fromInt", SCALAR_FACTORY_SIG("I"));
  if (Scalar_fromInt == nullptr) {
    return false;
  }
  Scalar_dateFromInt = env->GetStaticMethodID(cls, "dateFromInt", SCALAR_FACTORY_SIG("I"));
  if (Scalar_dateFromInt == nullptr) {
    return false;
  }
  Scalar_fromLong = env->GetStaticMethodID(cls, "fromLong", SCALAR_FACTORY_SIG("J"));
  if (Scalar_fromLong == nullptr) {
    return false;
  }
  Scalar_dateFromLong = env->GetStaticMethodID(cls, "dateFromLong", SCALAR_FACTORY_SIG("J"));
  if (Scalar_dateFromLong == nullptr) {
    return false;
  }
  Scalar_timestampFromLong =
      env->GetStaticMethodID(cls, "timestampFromLong", SCALAR_FACTORY_SIG("JI"));
  if (Scalar_timestampFromLong == nullptr) {
    return false;
  }
  Scalar_fromFloat = env->GetStaticMethodID(cls, "fromFloat", SCALAR_FACTORY_SIG("F"));
  if (Scalar_fromFloat == nullptr) {
    return false;
  }
  Scalar_fromDouble = env->GetStaticMethodID(cls, "fromDouble", SCALAR_FACTORY_SIG("D"));
  if (Scalar_fromDouble == nullptr) {
    return false;
  }

  // Convert local reference to global so it cannot be garbage collected.
  Scalar_jclass = static_cast<jclass>(env->NewGlobalRef(cls));
  if (Scalar_jclass == nullptr) {
    return false;
  }

  return true;
}

static void release_scalar_jni(JNIEnv *env) {
  if (Scalar_jclass != nullptr) {
    env->DeleteGlobalRef(Scalar_jclass);
    Scalar_jclass = nullptr;
  }
}

static jobject jscalar_from_scalar(JNIEnv *env, const gdf_scalar &scalar, gdf_time_unit time_unit) {
  jobject obj = nullptr;
  if (scalar.is_valid) {
    switch (scalar.dtype) {
      case GDF_INT8:
        obj = env->CallStaticObjectMethod(Scalar_jclass, Scalar_fromByte, scalar.data.si08);
        break;
      case GDF_INT16:
        obj = env->CallStaticObjectMethod(Scalar_jclass, Scalar_fromShort, scalar.data.si16);
        break;
      case GDF_INT32:
        obj = env->CallStaticObjectMethod(Scalar_jclass, Scalar_fromInt, scalar.data.si32);
        break;
      case GDF_INT64:
        obj = env->CallStaticObjectMethod(Scalar_jclass, Scalar_fromLong, scalar.data.si64);
        break;
      case GDF_FLOAT32:
        obj = env->CallStaticObjectMethod(Scalar_jclass, Scalar_fromFloat, scalar.data.fp32);
        break;
      case GDF_FLOAT64:
        obj = env->CallStaticObjectMethod(Scalar_jclass, Scalar_fromDouble, scalar.data.fp64);
        break;
      case GDF_BOOL8:
        obj = env->CallStaticObjectMethod(Scalar_jclass, Scalar_fromBool, scalar.data.b08);
        break;
      case GDF_DATE32:
        obj = env->CallStaticObjectMethod(Scalar_jclass, Scalar_dateFromInt, scalar.data.dt32);
        break;
      case GDF_DATE64:
        obj = env->CallStaticObjectMethod(Scalar_jclass, Scalar_dateFromLong, scalar.data.dt64);
        break;
      case GDF_TIMESTAMP:
        obj = env->CallStaticObjectMethod(Scalar_jclass, Scalar_timestampFromLong, scalar.data.tmst,
                                          time_unit);
        break;
      default:
        throwJavaException(env, "java/lang/UnsupportedOperationException",
                           "Unsupported native scalar type");
        break;
    }
  } else {
    if (scalar.dtype == GDF_TIMESTAMP) {
      obj = env->CallStaticObjectMethod(Scalar_jclass, Scalar_timestampFromNull, time_unit);
    } else {
      obj = env->CallStaticObjectMethod(Scalar_jclass, Scalar_fromNull, scalar.dtype);
    }
  }
  return obj;
}

static void gdf_scalar_init(gdf_scalar *scalar, jlong intValues, jfloat fValue, jdouble dValue,
                            jboolean isValid, int dType) {
  scalar->dtype = static_cast<gdf_dtype>(dType);
  scalar->is_valid = isValid;
  switch (scalar->dtype) {
    case GDF_INT8: scalar->data.si08 = static_cast<char>(intValues); break;
    case GDF_INT16: scalar->data.si16 = static_cast<short>(intValues); break;
    case GDF_INT32: scalar->data.si32 = static_cast<int>(intValues); break;
    case GDF_INT64: scalar->data.si64 = static_cast<long>(intValues); break;
    case GDF_DATE32: scalar->data.dt32 = static_cast<gdf_date32>(intValues); break;
    case GDF_DATE64: scalar->data.dt64 = static_cast<gdf_date64>(intValues); break;
    case GDF_TIMESTAMP: scalar->data.tmst = static_cast<gdf_timestamp>(intValues); break;
    case GDF_BOOL8: scalar->data.b08 = static_cast<char>(intValues); break;
    case GDF_FLOAT32: scalar->data.fp32 = fValue; break;
    case GDF_FLOAT64: scalar->data.fp64 = dValue; break;
    default: throw std::logic_error("Unsupported scalar type");
  }
}

static jni_rmm_unique_ptr<gdf_valid_type>
copy_validity(JNIEnv *env, gdf_size_type size, gdf_size_type null_count, gdf_valid_type *valid) {
  jni_rmm_unique_ptr<gdf_valid_type> ret{};
  if (null_count > 0) {
    gdf_size_type copy_size = ((size + 7) / 8);
    gdf_size_type alloc_size = gdf_valid_allocation_size(size);
    ret = jniRmmAlloc<gdf_valid_type>(env, alloc_size);
    JNI_CUDA_TRY(env, 0, cudaMemcpy(ret.get(), valid, copy_size, cudaMemcpyDeviceToDevice));
  }
  return ret;
}

static jlong cast_string_cat_to(JNIEnv *env, NVCategory *cat, gdf_dtype target_type,
                                gdf_time_unit target_unit, gdf_size_type size,
                                gdf_size_type null_count, gdf_valid_type *valid) {
  switch (target_type) {
    case GDF_STRING: {
      unique_nvstr_ptr str(cat->to_strings(), &NVStrings::destroy);

      jni_rmm_unique_ptr<gdf_valid_type> valid_copy = copy_validity(env, size, null_count, valid);

      gdf_column_wrapper output(size, target_type, null_count, str.release(), valid_copy.release());
      return reinterpret_cast<jlong>(output.release());
    }
    default: throw std::logic_error("Unsupported type to cast a string_cat to");
  }
}

static jlong cast_string_to(JNIEnv *env, NVStrings *str, gdf_dtype target_type,
                            gdf_time_unit target_unit, gdf_size_type size, gdf_size_type null_count,
                            gdf_valid_type *valid) {
  switch (target_type) {
    case GDF_STRING_CATEGORY: {
      unique_nvcat_ptr cat(NVCategory::create_from_strings(*str), &NVCategory::destroy);
      auto cat_data = jniRmmAlloc<int>(env, sizeof(int) * size);
      if (size != cat->get_values(cat_data.get(), true)) {
        JNI_THROW_NEW(env, "java/lang/IllegalStateException", "Internal Error copying str cat data",
                      0);
      }

      jni_rmm_unique_ptr<gdf_valid_type> valid_copy = copy_validity(env, size, null_count, valid);

      gdf_column_wrapper output(size, target_type, null_count, cat_data.release(),
                                valid_copy.release(), cat.release());
      return reinterpret_cast<jlong>(output.release());
    }
    default: throw std::logic_error("Unsupported type to cast a string to");
  }
}

} // namespace jni
} // namespace cudf

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *) {
  JNIEnv *env;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), cudf::jni::MINIMUM_JNI_VERSION) != JNI_OK) {
    return JNI_ERR;
  }

  // cache some class/method/field lookups
  if (!cudf::jni::cache_scalar_jni(env)) {
    return JNI_ERR;
  }

  return cudf::jni::MINIMUM_JNI_VERSION;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *) {
  JNIEnv *env = nullptr;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), cudf::jni::MINIMUM_JNI_VERSION) != JNI_OK) {
    return;
  }

  cudf::jni::release_scalar_jni(env);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfUnaryMath(JNIEnv *env, jclass, jlong inputPtr,
                                                              jint intOp, jint outDtype) {
  JNI_NULL_CHECK(env, inputPtr, "input is null", 0);
  try {
    gdf_column *input = reinterpret_cast<gdf_column *>(inputPtr);
    gdf_dtype out_type = static_cast<gdf_dtype>(outDtype);
    gdf_unary_math_op op = static_cast<gdf_unary_math_op>(intOp);
    cudf::jni::gdf_column_wrapper ret(input->size, out_type, input->null_count > 0);

    JNI_GDF_TRY(env, 0, gdf_unary_math(input, ret.get(), op));
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfBinaryOpVV(JNIEnv *env, jclass, jlong lhsPtr,
                                                               jlong rhsPtr, jint intOp,
                                                               jint outDtype) {
  JNI_NULL_CHECK(env, lhsPtr, "lhs is null", 0);
  JNI_NULL_CHECK(env, rhsPtr, "rhs is null", 0);
  try {
    gdf_column *lhs = reinterpret_cast<gdf_column *>(lhsPtr);
    gdf_column *rhs = reinterpret_cast<gdf_column *>(rhsPtr);
    gdf_dtype out_type = static_cast<gdf_dtype>(outDtype);
    gdf_binary_operator op = static_cast<gdf_binary_operator>(intOp);
    cudf::jni::gdf_column_wrapper ret(lhs->size, out_type,
                                      lhs->null_count > 0 || rhs->null_count > 0);

    JNI_GDF_TRY(env, 0, gdf_binary_operation_v_v(ret.get(), lhs, rhs, op));
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfBinaryOpSV(
    JNIEnv *env, jclass, jlong lhsIntValues, jfloat lhsFValue, jdouble lhsDValue,
    jboolean lhsIsValid, int lhsDType, jlong rhsPtr, jint intOp, jint outDtype) {
  JNI_NULL_CHECK(env, rhsPtr, "rhs is null", 0);
  try {
    gdf_scalar lhs{};
    cudf::jni::gdf_scalar_init(&lhs, lhsIntValues, lhsFValue, lhsDValue, lhsIsValid, lhsDType);
    gdf_column *rhs = reinterpret_cast<gdf_column *>(rhsPtr);
    gdf_dtype out_type = static_cast<gdf_dtype>(outDtype);
    gdf_binary_operator op = static_cast<gdf_binary_operator>(intOp);
    cudf::jni::gdf_column_wrapper ret(rhs->size, out_type, !lhs.is_valid || rhs->null_count > 0);

    JNI_GDF_TRY(env, 0, gdf_binary_operation_s_v(ret.get(), &lhs, rhs, op));
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfBinaryOpVS(JNIEnv *env, jclass, jlong lhsPtr,
                                                               jlong rhsIntValues, jfloat rhsFValue,
                                                               jdouble rhsDValue,
                                                               jboolean rhsIsValid, int rhsDType,
                                                               jint intOp, jint outDtype) {
  JNI_NULL_CHECK(env, lhsPtr, "lhs is null", 0);
  try {
    gdf_column *lhs = reinterpret_cast<gdf_column *>(lhsPtr);
    gdf_scalar rhs{};
    cudf::jni::gdf_scalar_init(&rhs, rhsIntValues, rhsFValue, rhsDValue, rhsIsValid, rhsDType);
    gdf_dtype out_type = static_cast<gdf_dtype>(outDtype);
    gdf_binary_operator op = static_cast<gdf_binary_operator>(intOp);
    cudf::jni::gdf_column_wrapper ret(lhs->size, out_type, !rhs.is_valid || lhs->null_count > 0);

    JNI_GDF_TRY(env, 0, gdf_binary_operation_v_s(ret.get(), lhs, &rhs, op));
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfExtractDatetimeYear(JNIEnv *env, jclass,
                                                                        jlong inputPtr) {
  JNI_NULL_CHECK(env, inputPtr, "input is null", 0);
  try {
    gdf_column *input = reinterpret_cast<gdf_column *>(inputPtr);
    cudf::jni::gdf_column_wrapper output(input->size, GDF_INT16, input->null_count != 0);
    JNI_GDF_TRY(env, 0, gdf_extract_datetime_year(input, output.get()));
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfExtractDatetimeMonth(JNIEnv *env, jclass,
                                                                         jlong inputPtr) {
  JNI_NULL_CHECK(env, inputPtr, "input is null", 0);
  try {
    gdf_column *input = reinterpret_cast<gdf_column *>(inputPtr);
    cudf::jni::gdf_column_wrapper output(input->size, GDF_INT16, input->null_count != 0);
    JNI_GDF_TRY(env, 0, gdf_extract_datetime_month(input, output.get()));
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfExtractDatetimeDay(JNIEnv *env, jclass,
                                                                       jlong inputPtr) {
  JNI_NULL_CHECK(env, inputPtr, "input is null", 0);
  try {
    gdf_column *input = reinterpret_cast<gdf_column *>(inputPtr);
    cudf::jni::gdf_column_wrapper output(input->size, GDF_INT16, input->null_count != 0);
    JNI_GDF_TRY(env, 0, gdf_extract_datetime_day(input, output.get()));
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfExtractDatetimeHour(JNIEnv *env, jclass,
                                                                        jlong inputPtr) {
  JNI_NULL_CHECK(env, inputPtr, "input is null", 0);
  try {
    gdf_column *input = reinterpret_cast<gdf_column *>(inputPtr);
    cudf::jni::gdf_column_wrapper output(input->size, GDF_INT16, input->null_count != 0);
    JNI_GDF_TRY(env, 0, gdf_extract_datetime_hour(input, output.get()));
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfExtractDatetimeMinute(JNIEnv *env, jclass,
                                                                          jlong inputPtr) {
  JNI_NULL_CHECK(env, inputPtr, "input is null", 0);
  try {
    gdf_column *input = reinterpret_cast<gdf_column *>(inputPtr);
    cudf::jni::gdf_column_wrapper output(input->size, GDF_INT16, input->null_count != 0);
    JNI_GDF_TRY(env, 0, gdf_extract_datetime_minute(input, output.get()));
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfExtractDatetimeSecond(JNIEnv *env, jclass,
                                                                          jlong inputPtr) {
  JNI_NULL_CHECK(env, inputPtr, "input is null", 0);
  try {
    gdf_column *input = reinterpret_cast<gdf_column *>(inputPtr);
    cudf::jni::gdf_column_wrapper output(input->size, GDF_INT16, input->null_count != 0);
    JNI_GDF_TRY(env, 0, gdf_extract_datetime_second(input, output.get()));
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfCast(JNIEnv *env, jclass, jlong inputPtr,
                                                         jint dtype, jint timeUnit) {
  JNI_NULL_CHECK(env, inputPtr, "input is null", 0);
  try {
    gdf_column *input = reinterpret_cast<gdf_column *>(inputPtr);
    gdf_dtype cDType = static_cast<gdf_dtype>(dtype);
    gdf_time_unit time_unit = static_cast<gdf_time_unit>(timeUnit);
    size_t size = input->size;
    if (input->dtype == GDF_STRING) {
      NVStrings *str = static_cast<NVStrings *>(input->data);
      return cudf::jni::cast_string_to(env, str, cDType, time_unit, size, input->null_count,
                                       input->valid);
    } else if (input->dtype == GDF_STRING_CATEGORY && cDType == GDF_STRING) {
      NVCategory *cat = static_cast<NVCategory *>(input->dtype_info.category);
      return cudf::jni::cast_string_cat_to(env, cat, cDType, time_unit, size, input->null_count,
                                           input->valid);
    } else {
      cudf::jni::gdf_column_wrapper output(input->size, cDType, input->null_count != 0);
      output.get()->dtype_info.time_unit = time_unit;
      JNI_GDF_TRY(env, 0, gdf_cast(input, output.get()));
      return reinterpret_cast<jlong>(output.release());
    }
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jobject JNICALL Java_ai_rapids_cudf_Cudf_reduction(JNIEnv *env, jclass, jlong jcol,
                                                             jint jop, jint jdtype) {
  try {
    gdf_column *col = reinterpret_cast<gdf_column *>(jcol);
    gdf_reduction_op op = static_cast<gdf_reduction_op>(jop);
    gdf_dtype dtype = static_cast<gdf_dtype>(jdtype);
    gdf_scalar scalar = cudf::reduction(col, op, dtype);
    return cudf::jni::jscalar_from_scalar(env, scalar, col->dtype_info.time_unit);
  }
  CATCH_STD(env, 0);
}

} // extern "C"
