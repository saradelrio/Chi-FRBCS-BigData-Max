/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.classifier.chi_rw.mapreduce;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.chi_rw.RuleBase;
import org.apache.mahout.classifier.chi_rw.Chi_RWUtils;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

/**
 * Used by various implementation to return the results of a build.<br>
 * Contains a rule base and and its predictions.
 */
public class MapredOutput implements Writable, Cloneable {

  private RuleBase rulebase;

  private int[] predictions;

  public MapredOutput() {
  }

  public MapredOutput(RuleBase rulebase, int[] predictions) {
    this.rulebase = rulebase;
    this.predictions = predictions;
  }

  public MapredOutput(RuleBase rulebase) {
    this(rulebase, null);
  }

  public RuleBase getRuleBase() {
    return rulebase;
  }

  int[] getPredictions() {
    return predictions;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    boolean readRuleBase = in.readBoolean();
    if (readRuleBase) {
      RuleBase rb = new RuleBase();
      rb.readFields(in);
      rulebase = rb;	
    }

    boolean readPredictions = in.readBoolean();
    if (readPredictions) {
      predictions = Chi_RWUtils.readIntArray(in);
    }
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeBoolean(rulebase != null);
    if (rulebase != null) {
      rulebase.write(out);
    }

    out.writeBoolean(predictions != null);
    if (predictions != null) {
      Chi_RWUtils.writeArray(out, predictions);
    }
  }

  @Override
  public MapredOutput clone() {
    return new MapredOutput(rulebase, predictions);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof MapredOutput)) {
      return false;
    }

    MapredOutput mo = (MapredOutput) obj;

    return ((rulebase == null && mo.getRuleBase() == null) || (rulebase != null && rulebase.equals(mo.getRuleBase())))
        && Arrays.equals(predictions, mo.getPredictions());
  }

  @Override
  public int hashCode() {
    int hashCode = rulebase == null ? 1 : rulebase.hashCode();
    for (int prediction : predictions) {
      hashCode = 31 * hashCode + prediction;
    }
    return hashCode;
  }

  @Override
  public String toString() {
    return "{" + rulebase + " | " + Arrays.toString(predictions) + '}';
  }

}

