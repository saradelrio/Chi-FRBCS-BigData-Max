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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.mahout.classifier.chi_rw.Chi_RWUtils;
import org.apache.mahout.classifier.chi_rw.RuleBase;
import org.apache.mahout.classifier.chi_rw.builder.Fuzzy_ChiBuilder;
import org.apache.mahout.classifier.chi_rw.mapreduce.*;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;

import java.io.IOException;

/**
 * Builds a model using partial data. Each mapper uses only the data given by its InputSplit
 */
public class PartialBuilder extends Builder {

  public PartialBuilder(Fuzzy_ChiBuilder fuzzy_ChiBuilder, Path dataPath, Path datasetPath) {
    this(fuzzy_ChiBuilder, dataPath, datasetPath, new Configuration());
  }
  
  public PartialBuilder(Fuzzy_ChiBuilder fuzzy_ChiBuilder,
                        Path dataPath,
                        Path datasetPath,
                        Configuration conf) {
    super(fuzzy_ChiBuilder, dataPath, datasetPath, conf);
  }

  @Override
  protected void configureJob(Job job) throws IOException {
    Configuration conf = job.getConfiguration();
    
    job.setJarByClass(PartialBuilder.class);
    
    FileInputFormat.setInputPaths(job, getDataPath());
    FileOutputFormat.setOutputPath(job, getOutputPath(conf));
    
    job.setMapOutputKeyClass(LongWritable.class);
    job.setMapOutputValueClass(MapredOutput.class);
    
    job.setOutputKeyClass(LongWritable.class);
    job.setOutputValueClass(RuleBase.class);
    
    job.setMapperClass(ChiMapper.class);
    job.setReducerClass(ChiReducer.class);
    
    job.setNumReduceTasks(1);
    
    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
  }

  @Override
  protected RuleBase parseOutput(Job job) throws IOException {
    Configuration conf = job.getConfiguration();
    
    Path outputPath = getOutputPath(conf);
      
    return processOutput(job, outputPath);
  }
  
  protected RuleBase processOutput(JobContext job, Path outputPath) throws IOException {
    	  
    Configuration conf = job.getConfiguration();

    FileSystem fs = outputPath.getFileSystem(conf);

    Path[] outfiles = Chi_RWUtils.listOutputFiles(fs, outputPath);
    
    RuleBase ruleBase = null;
    
    // read all the outputs
    for (Path path : outfiles) {
      for (Pair<LongWritable,RuleBase> record : new SequenceFileIterable<LongWritable, RuleBase>(path, conf)) {
    	if(ruleBase == null){
          ruleBase = record.getSecond();
    	}
      }
    }
    
    return ruleBase;
  }
  
}

