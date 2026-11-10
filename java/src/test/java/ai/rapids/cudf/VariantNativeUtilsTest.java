/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import ai.rapids.cudf.HostColumnVector.BasicType;
import ai.rapids.cudf.HostColumnVector.ListType;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class VariantNativeUtilsTest extends CudfTestBase {
  private static final ListType BINARY = new ListType(true, new BasicType(false, DType.UINT8));

  private static List<Byte> integer(int width, long value) {
    int type = width == 1 ? 3 : width == 2 ? 4 : width == 4 ? 5 : 6;
    List<Byte> bytes = new ArrayList<>();
    bytes.add((byte) (type << 2));
    for (int i = 0; i < width; i++) {
      bytes.add((byte) (value >>> (8 * i)));
    }
    return bytes;
  }

  @Test
  void coverageCombinesDecodersPerRowAndKeepsOriginalNullMeaning() {
    try (ColumnVector original = ColumnVector.fromUnsignedBytes((byte) 0, (byte) 0, (byte) 1);
         ColumnVector first = ColumnVector.fromUnsignedBytes((byte) 0, (byte) 4, (byte) 4);
         ColumnVector second = ColumnVector.fromUnsignedBytes((byte) 4, (byte) 0, (byte) 4);
         ColumnVector decodedNulls = ColumnVector.fromUnsignedBytes((byte) 1, (byte) 1, (byte) 1)) {
      assertTrue(VariantUtils.allVariantRowsCovered(original, new ColumnView[]{first, second}));
      assertFalse(VariantUtils.allVariantRowsCovered(original, new ColumnView[]{first}));
      assertFalse(VariantUtils.allVariantRowsCovered(original, new ColumnView[]{decodedNulls}));
      assertFalse(VariantUtils.allVariantRowsCovered(original, new ColumnView[]{}));
      try (ColumnVector expected = ColumnVector.fromUnsignedBytes((byte) 0, (byte) 0, (byte) 1)) {
        assertColumnsAreEqual(expected, original);
      }
    }
  }

  @Test
  void coverageHandlesArbitraryDecoderCountSlicesAndEmpty() {
    byte[] originalBytes = new byte[257];
    byte[][] decoderBytes = new byte[7][257];
    Random random = new Random(421907);
    for (int row = 0; row < originalBytes.length; row++) {
      originalBytes[row] = (byte) random.nextInt(8);
      for (byte[] decoder : decoderBytes) {
        decoder[row] = (byte) (1 + random.nextInt(7));
      }
      decoderBytes[random.nextInt(decoderBytes.length)][row] = 0;
    }
    List<ColumnVector> decoders = new ArrayList<>();
    try (ColumnVector original = ColumnVector.fromUnsignedBytes(originalBytes);
         ColumnVector empty = ColumnVector.fromUnsignedBytes()) {
      try {
        for (byte[] decoder : decoderBytes) {
          decoders.add(ColumnVector.fromUnsignedBytes(decoder));
        }
        assertTrue(VariantUtils.allVariantRowsCovered(original,
            decoders.toArray(new ColumnView[0])));
      } finally {
        for (ColumnVector decoder : decoders) {
          decoder.close();
        }
      }
      assertTrue(VariantUtils.allVariantRowsCovered(empty, new ColumnView[]{}));
      assertTrue(VariantUtils.allVariantRowsCovered(empty, new ColumnView[]{empty}));
    }
    try (ColumnVector original = ColumnVector.fromUnsignedBytes(
             (byte) 4, (byte) 0, (byte) 1, (byte) 0, (byte) 4);
         ColumnVector decoded = ColumnVector.fromUnsignedBytes(
             (byte) 4, (byte) 0, (byte) 4, (byte) 0, (byte) 4);
         CloseableArray<ColumnView> originalViews = CloseableArray.wrap(original.splitAsViews(1, 4));
         CloseableArray<ColumnView> decodedViews = CloseableArray.wrap(decoded.splitAsViews(1, 4))) {
      assertTrue(VariantUtils.allVariantRowsCovered(originalViews.get(1),
          new ColumnView[]{decodedViews.get(1)}));
      assertFalse(VariantUtils.allVariantRowsCovered(original, new ColumnView[]{decoded}));
    }
  }

  @Test
  void integerDecodeMatchesFourExactDecodersForHeadersPayloadsAndRandomWidths() {
    List<List<Byte>> rows = new ArrayList<>();
    rows.add(null);
    rows.add(new ArrayList<>());
    for (int header = 0; header < 256; header++) {
      for (int length = 1; length <= 10; length++) {
        List<Byte> bytes = new ArrayList<>();
        bytes.add((byte) header);
        for (int i = 1; i < length; i++) {
          bytes.add((byte) (i * 37));
        }
        rows.add(bytes);
      }
    }
    Random random = new Random(9172026);
    for (int width : new int[]{1, 2, 4, 8}) {
      long min = width == 8 ? Long.MIN_VALUE : -(1L << (8 * width - 1));
      long max = width == 8 ? Long.MAX_VALUE : (1L << (8 * width - 1)) - 1;
      for (long value : new long[]{min, max, -1, 0, 1}) {
        rows.add(integer(width, value));
      }
      for (int i = 0; i < 257; i++) {
        rows.add(integer(width, random.nextLong()));
      }
    }
    @SuppressWarnings("unchecked")
    List<Byte>[] encoded = rows.toArray(new List[0]);
    byte[] priorBytes = new byte[encoded.length];
    try (ColumnVector values = ColumnVector.fromLists(BINARY, encoded);
         ColumnVector prior = ColumnVector.fromUnsignedBytes(priorBytes)) {
      assertIntegerDifferential(values, prior);
      try (CloseableArray<ColumnView> valueViews = CloseableArray.wrap(values.splitAsViews(1, 253));
           CloseableArray<ColumnView> statusViews = CloseableArray.wrap(prior.splitAsViews(1, 253))) {
        assertIntegerDifferential(valueViews.get(1), statusViews.get(1));
      }
    }
  }

  private static void assertIntegerDifferential(ColumnView values, ColumnView prior) {
    int rows = (int) values.getRowCount();
    Long[] expectedValues = new Long[rows];
    byte[] expectedStatuses = new byte[rows];
    boolean first = true;
    for (DType type : new DType[]{DType.INT8, DType.INT16, DType.INT32, DType.INT64}) {
      try (VariantOperationResult exact = VariantUtils.castVariantValueWithStatus(values, type, prior);
           ColumnVector wide = exact.getValue().castTo(DType.INT64);
           HostColumnVector hostValues = wide.copyToHost();
           HostColumnVector hostStatuses = exact.getStatus().copyToHost()) {
        for (int row = 0; row < rows; row++) {
          byte status = hostStatuses.getByte(row);
          if (first || status == 0 || (expectedStatuses[row] != 0 && status == 5)) {
            expectedStatuses[row] = status;
          }
          if (status == 0) {
            expectedValues[row] = hostValues.getLong(row);
          }
        }
      }
      first = false;
    }
    try (ColumnVector before = prior.copyToColumnVector();
         VariantOperationResult unified = VariantUtils.castVariantIntegerToLongWithStatus(values, prior);
         ColumnVector expected = ColumnVector.fromBoxedLongs(expectedValues);
         ColumnVector expectedStatus = ColumnVector.fromUnsignedBytes(expectedStatuses)) {
      assertColumnsAreEqual(expected, unified.getValue());
      assertColumnsAreEqual(expectedStatus, unified.getStatus());
      assertColumnsAreEqual(before, prior);
      assertNull(unified.getStatus().getValid());
    }
  }

  @Test
  void integerDecodePreservesEveryPriorStatusAndNulls() {
    @SuppressWarnings("unchecked")
    List<Byte>[] rows = new List[256];
    byte[] status = new byte[rows.length];
    for (int row = 0; row < rows.length; row++) {
      rows[row] = integer(8, Long.MIN_VALUE);
      status[row] = (byte) row;
    }
    try (ColumnVector values = ColumnVector.fromLists(BINARY, rows);
         ColumnVector prior = ColumnVector.fromUnsignedBytes(status)) {
      assertIntegerDifferential(values, prior);
    }
    try (ColumnVector values = ColumnVector.fromLists(BINARY, null, null, null);
         ColumnVector prior = ColumnVector.fromUnsignedBytes((byte) 0, (byte) 1, (byte) 2)) {
      assertIntegerDifferential(values, prior);
      try (VariantOperationResult decoded = VariantUtils.castVariantIntegerToLongWithStatus(values, prior)) {
        assertFalse(VariantUtils.allVariantRowsCovered(prior, new ColumnView[]{decoded.getStatus()}));
      }
    }
    try (ColumnVector values = ColumnVector.fromLists(BINARY);
         ColumnVector prior = ColumnVector.fromUnsignedBytes();
         VariantOperationResult decoded = VariantUtils.castVariantIntegerToLongWithStatus(values, prior);
         ColumnVector expected = ColumnVector.fromBoxedLongs()) {
      assertColumnsAreEqual(expected, decoded.getValue());
      assertColumnsAreEqual(prior, decoded.getStatus());
    }
  }

  @Test
  void validationRejectsMasksTypesLengthsAndNullArgumentsBeforeEmptyFastPath() {
    try (ColumnVector values = ColumnVector.fromLists(BINARY, integer(1, 1));
         ColumnVector original = ColumnVector.fromUnsignedBytes((byte) 0);
         ColumnVector empty = ColumnVector.fromUnsignedBytes();
         ColumnVector wrongType = ColumnVector.fromInts(0);
         ColumnVector nullable = ColumnVector.fromBoxedUnsignedBytes((byte) 0, null);
         CloseableArray<ColumnView> allValidMasked = CloseableArray.wrap(nullable.splitAsViews(1))) {
      assertThrows(IllegalArgumentException.class,
          () -> VariantUtils.allVariantRowsCovered(wrongType, new ColumnView[]{}));
      assertThrows(IllegalArgumentException.class,
          () -> VariantUtils.allVariantRowsCovered(original, new ColumnView[]{empty}));
      assertThrows(IllegalArgumentException.class,
          () -> VariantUtils.allVariantRowsCovered(empty, new ColumnView[]{original}));
      assertThrows(IllegalArgumentException.class,
          () -> VariantUtils.allVariantRowsCovered(original, new ColumnView[]{allValidMasked.get(0)}));
      assertThrows(IllegalArgumentException.class,
          () -> VariantUtils.castVariantIntegerToLongWithStatus(values, allValidMasked.get(0)));
      assertThrows(IllegalArgumentException.class,
          () -> VariantUtils.castVariantIntegerToLongWithStatus(values, wrongType));
      assertThrows(IllegalArgumentException.class,
          () -> VariantUtils.castVariantIntegerToLongWithStatus(values, empty));
      assertThrows(CudfException.class,
          () -> VariantUtils.castVariantIntegerToLongWithStatus(original, original));
      assertThrows(NullPointerException.class,
          () -> VariantUtils.allVariantRowsCovered(null, new ColumnView[]{}));
      assertThrows(NullPointerException.class,
          () -> VariantUtils.allVariantRowsCovered(original, null));
      assertThrows(NullPointerException.class,
          () -> VariantUtils.allVariantRowsCovered(original, new ColumnView[]{null}));
      assertThrows(NullPointerException.class,
          () -> VariantUtils.castVariantIntegerToLongWithStatus(null, original));
      assertThrows(NullPointerException.class,
          () -> VariantUtils.castVariantIntegerToLongWithStatus(values, null));
    }
    ListType wrongChild = new ListType(true, new BasicType(false, DType.INT32));
    try (ColumnVector values = ColumnVector.fromLists(wrongChild, Arrays.asList(1));
         ColumnVector status = ColumnVector.fromUnsignedBytes((byte) 0)) {
      assertThrows(CudfException.class,
          () -> VariantUtils.castVariantIntegerToLongWithStatus(values, status));
    }
  }
}
