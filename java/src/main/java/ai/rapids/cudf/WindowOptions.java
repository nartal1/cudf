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
 * Arguments for Window function (see cudf::rolling_window)
 * windowSize, minPeriods and forwardWindow is used for creating static sizes
 * for all the rows.
 * windowCol, minPeriodsCol and forwardWindowCol is used to support dynamic window.
 * i.e each row will have it's own windowsize, minPeriods and forwardWindow.
 * Caller is responsible for the lifecycle of the vectors.
 */

public class WindowOptions {

  public static WindowOptions DEFAULT = new WindowOptions(new Builder());

  private final int windowSize;
  private final int minPeriods;
  private final int forwardWindow;
  private final AggregateOp aggType;
  private final ColumnVector windowCol;
  private final ColumnVector minPeriodsCol;
  private final ColumnVector forwardWindowCol;


  private WindowOptions(Builder builder) {
    this.windowSize = builder.windowSize;
    this.minPeriods = builder.minPeriods;
    this.forwardWindow = builder.forwardWindow;
    this.aggType = builder.aggType;
    this.windowCol = builder.windowCol;
    this.minPeriodsCol = builder.minPeriodsCol;
    this.forwardWindowCol = builder.forwardWindowCol;
  }

  public static Builder builder(){
    return new Builder();
  }

  int getWindow() { return this.windowSize; }

  int getMinPeriods() { return  this.minPeriods; }

  int getForwardWindow() { return this.forwardWindow; }

  AggregateOp getAggType() { return this.aggType; }

  ColumnVector getWindowCol() { return  windowCol; }

  ColumnVector getMinPeriodsCol() { return  this.minPeriodsCol; }

  ColumnVector getForwardWindowCol() { return this.forwardWindowCol; }

  public static class Builder {
    private int windowSize = -1;
    private int minPeriods = -1;
    private int forwardWindow = -1;
    private AggregateOp aggType = AggregateOp.SUM;
    private ColumnVector windowCol = null;
    private ColumnVector minPeriodsCol = null;
    private ColumnVector forwardWindowCol = null;

    /**
     * Set the static rolling window size.
     */
    public Builder windowSize(int windowSize) {
      if (windowSize < 0 ) {
        throw  new IllegalArgumentException("Window size must be non negative");
      }
      this.windowSize = windowSize;
      return this;
    }

    /**
     * Set the static minimum number of observation required to evaluate element.
     */
    public Builder minPeriods(int minPeriods) {
      if (minPeriods < 0 ) {
        throw  new IllegalArgumentException("Minimum observations must be non negative");
      }
      this.minPeriods = minPeriods;
      return this;
    }

    /**
     * Set the static window size in forward direction.
     */
    public Builder forwardWindow(int forwardWindow) {
      if (forwardWindow < 0 ) {
        throw  new IllegalArgumentException("Forward window size must be non negative");
      }
      this.forwardWindow = forwardWindow;
      return this;
    }

    /**
     * Set the rolling window aggregation type.
     */

    public Builder aggType(AggregateOp aggType) {
      if (aggType.nativeId < 0 || aggType.nativeId > 4) {
        throw new IllegalArgumentException("Invalid Aggregation Type");
      }
      this.aggType = aggType;
      return this;
    }

    /**
     * Set the window size values for each element in the column.
     * The caller owns the vector which is passed in below and is responsible for
     * it's lifecycle.
     */
    public Builder windowCol(ColumnVector windowCol){
      this.windowCol = windowCol;
      return this;
    }

    /**
     * Set the minimum number of observations for each element in the column.
     * The caller owns the vector which is passed in below and is responsible for
     * it's lifecycle.
     */
    public Builder minPeriodsCol(ColumnVector minPeriodsCol){
      this.minPeriodsCol = minPeriodsCol;
      return this;
    }

    /**
     * Set the forward window size values for each element in the column.
     * The caller owns the vector which is passed in below and is responsible for
     * it's lifecycle.
     */
    public Builder forwardWindowCol(ColumnVector forwardWindowCol){
      this.forwardWindowCol = forwardWindowCol;
      return this;
    }

    public WindowOptions build() {
      if (windowCol != null && windowSize != -1)
        throw new IllegalArgumentException("Either windowSize or windCol should be provided");
      if (minPeriodsCol != null && minPeriods != -1)
        throw new IllegalArgumentException("Either minPeriods or minPeriodsCol should be provided");
      if (forwardWindowCol != null && forwardWindow != -1)
        throw new IllegalArgumentException
          ("Either forwardWindow or forwardWindowCol should be provided");
      return new WindowOptions(this);
    }
  }
}