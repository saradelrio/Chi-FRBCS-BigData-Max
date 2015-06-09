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
package org.apache.mahout.classifier.chi_rwcs.mapreduce;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.classifier.chi_rwcs.builder.Fuzzy_ChiCSBuilder;
import org.apache.mahout.classifier.chi_rwcs.data.Dataset;

import java.io.IOException;

/**
 * Base class for Mapred mappers. Loads common parameters from the job
 */
public class MapredMapper<KEYIN,VALUEIN,KEYOUT,VALUEOUT> extends Mapper<KEYIN,VALUEIN,KEYOUT,VALUEOUT> {
  
  private boolean noOutput;
  
  protected Fuzzy_ChiCSBuilder fuzzy_ChiCSBuilder;
  
  private Dataset dataset;
  
  /**
   * 
   * @return if false, the mapper does not estimate and output predictions
   */
  protected boolean isNoOutput() {
    return noOutput;
  }
  
  protected Fuzzy_ChiCSBuilder getFuzzy_ChiCSBuilder() {
    return fuzzy_ChiCSBuilder;
  }
  
  protected Dataset getDataset() {
    return dataset;
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    
    configure(!Builder.isOutput(conf), Builder.getFuzzy_ChiCSBuilder(conf), Builder.loadDataset(conf));
  }
  
  /**
   * Useful for testing
   */
  protected void configure(boolean noOutput, Fuzzy_ChiCSBuilder fuzzy_ChiCSBuilder, Dataset dataset) {
    Preconditions.checkArgument(fuzzy_ChiCSBuilder != null, "Fuzzy_ChiCSBuilder not found in the Job parameters");
    this.noOutput = noOutput;
    this.fuzzy_ChiCSBuilder = fuzzy_ChiCSBuilder;
    this.dataset = dataset;
  }
}

