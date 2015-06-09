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
package org.apache.mahout.classifier.chi_rwcs.mapreduce.partial;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.mahout.classifier.chi_rwcs.RuleBase;
import org.apache.mahout.classifier.chi_rwcs.mapreduce.MapredOutput;
import org.apache.mahout.classifier.chi_rwcs.mapreduce.Builder;
import org.apache.mahout.classifier.chi_rwcs.mapreduce.MapredMapper;
import org.apache.mahout.classifier.chi_rwcs.data.Data;
import org.apache.mahout.classifier.chi_rwcs.data.DataConverter;
import org.apache.mahout.classifier.chi_rwcs.data.Dataset;
import org.apache.mahout.classifier.chi_rwcs.data.Instance;
import org.apache.mahout.classifier.chi_rwcs.data.DataLoader;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;

public class ChiCSMapper extends MapredMapper<LongWritable,Text,LongWritable,MapredOutput> {
  
  private static final Logger log = LoggerFactory.getLogger(ChiCSMapper.class);
  
  /** used to convert input values to data instances */
  private DataConverter converter;
  
  /**first id */
  private int firstId = 0;
  
  /** mapper's partition */
  private int partition;
  
  /** will contain all instances if this mapper's split */
  private final List<Instance> instances = Lists.newArrayList();
  
  //Costs associated to each class elements
  int positive_class; // Which is the positive class
  double negative_class_cost = 1.0;
  double positive_class_cost;    
  
  public int getFirstTreeId() {
    return firstId;
  }
  
  /**
   * Load the training data
   */
  private static Data loadData(Configuration conf, Dataset dataset) throws IOException {
    Path dataPath = Builder.getDistributedCacheFile(conf, 1);
    FileSystem fs = FileSystem.get(dataPath.toUri(), conf);
    return DataLoader.loadData(dataset, fs, dataPath);
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    
    log.info("Loading the data...");
    Data data = loadData(conf, getDataset());
    log.info("Data loaded : {} instances", data.size());
    
    configure(conf.getInt("mapred.task.partition", -1), Builder.getNumMaps(conf), data);
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
  protected void configure(int partition, int numMapTasks, Data data) {
    converter = new DataConverter(getDataset());

    // mapper's partition
    Preconditions.checkArgument(partition >= 0, "Wrong partition ID");
    this.partition = partition;
    
    // Compute the distribution for all classes
    int classes_distribution [] = data.computeClassDistribution();
    
    // Compute the costs for the classes
    positive_class = data.computePositiveClass(classes_distribution);
    
    // Compute the costs associated to the class
    positive_class_cost = data.computePositiveClassCost(classes_distribution, positive_class);
    
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
        
    fuzzy_ChiCSBuilder.build(data, positive_class, positive_class_cost, negative_class_cost, context);    
    
    RuleBase ruleBase = fuzzy_ChiCSBuilder.getRuleBase();
    
    LongWritable key = new LongWritable(1);
      
    if (!isNoOutput()) {
      MapredOutput emOut = new MapredOutput(ruleBase);
      context.write(key, emOut);
    }
  }
}
