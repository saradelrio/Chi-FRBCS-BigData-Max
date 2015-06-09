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
package org.apache.mahout.classifier.chi_rw.mapreduce.partial;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.mahout.classifier.chi_rw.RuleBase;
import org.apache.mahout.classifier.chi_rw.mapreduce.MapredOutput;
import org.apache.mahout.classifier.chi_rw.mapreduce.Builder;
import org.apache.mahout.classifier.chi_rw.mapreduce.MapredMapper;
import org.apache.mahout.classifier.chi_rw.data.Data;
import org.apache.mahout.classifier.chi_rw.data.DataConverter;
import org.apache.mahout.classifier.chi_rw.data.Instance;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;

public class ChiMapper extends MapredMapper<LongWritable,Text,LongWritable,MapredOutput> {
  
  private static final Logger log = LoggerFactory.getLogger(ChiMapper.class);
  
  /** used to convert input values to data instances */
  private DataConverter converter;
  
  /**first id */
  private int firstId = 0;
  
  /** mapper's partition */
  private int partition;
  
  /** will contain all instances if this mapper's split */
  private final List<Instance> instances = Lists.newArrayList();
  
  public int getFirstTreeId() {
    return firstId;
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    
    configure(conf.getInt("mapred.task.partition", -1), Builder.getNumMaps(conf));
  }
  
  /**
   * Useful when testing
   * 
   * @param partition
   *          current mapper inputSplit partition
   * @param numMapTasks
   *          number of running map tasks
   * @param numTrees
   *          total number of trees in the forest
   */
  protected void configure(int partition, int numMapTasks) {
    converter = new DataConverter(getDataset());

    // mapper's partition
    Preconditions.checkArgument(partition >= 0, "Wrong partition ID");
    this.partition = partition;
    
    log.debug("partition : {}", partition);
  }
  
  @Override
  protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
    instances.add(converter.convert(value.toString()));
  }
  
  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
    // prepare the data
    log.debug("partition: {} numInstances: {}", partition, instances.size());
    
    Data data = new Data(getDataset(), instances);
    
    fuzzy_ChiBuilder.build(data, context);
    
    RuleBase ruleBase = fuzzy_ChiBuilder.getRuleBase();
    
    LongWritable key = new LongWritable(1);
      
    if (!isNoOutput()) {
      MapredOutput emOut = new MapredOutput(ruleBase);
      context.write(key, emOut);
    }
  }
}
