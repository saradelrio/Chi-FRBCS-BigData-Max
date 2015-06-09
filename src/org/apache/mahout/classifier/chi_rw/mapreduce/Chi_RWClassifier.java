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

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.classifier.chi_rw.Chi_RWUtils;
import org.apache.mahout.classifier.chi_rw.RuleBase;
import org.apache.mahout.classifier.chi_rw.data.DataConverter;
import org.apache.mahout.classifier.chi_rw.data.Dataset;
import org.apache.mahout.classifier.chi_rw.data.Instance;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Mapreduce implementation that classifies the Input data using a previousely built model
 */
public class Chi_RWClassifier {
	
  private static final Logger log = LoggerFactory.getLogger(Chi_RWClassifier.class);
  private final Path modelPath;
  private final Path inputPath;
  private final Path datasetPath;
  private final Configuration conf;
  private final Path outputPath; // path that will containt the final output of the classifier
  private final Path mappersOutputPath; // mappers will output here
  private double[][] results;
	  
  public double[][] getResults() {
    return results;
  }
  
  public Chi_RWClassifier(Path modelPath, Path inputPath, Path datasetPath, Path outputPath, Configuration conf) {
    this.modelPath = modelPath;
    this.inputPath = inputPath;
    this.datasetPath = datasetPath;
    this.outputPath = outputPath;
    this.conf = conf;
    mappersOutputPath = new Path(outputPath, "mappers");
  }
  
  private void configureJob(Job job) throws IOException {
    job.setJarByClass(Chi_RWClassifier.class);

	FileInputFormat.setInputPaths(job, inputPath);
	FileOutputFormat.setOutputPath(job, mappersOutputPath);

	job.setOutputKeyClass(DoubleWritable.class);
	job.setOutputValueClass(Text.class);

	job.setMapperClass(ClassifierMapper.class);
	job.setNumReduceTasks(0); // no reducers

	job.setInputFormatClass(ClassifierTextInputFormat.class);
	job.setOutputFormatClass(SequenceFileOutputFormat.class);
  }

  public void run() throws IOException, ClassNotFoundException, InterruptedException {
    
	FileSystem fs = FileSystem.get(conf);

	// check the output
	if (fs.exists(outputPath)) {
	  throw new IOException("Chi: Output path already exists : " + outputPath);
	}

	log.info("Chi: Adding the dataset to the DistributedCache");
	// put the dataset into the DistributedCache
	DistributedCache.addCacheFile(datasetPath.toUri(), conf);

	log.info("Chi: Adding the model to the DistributedCache");
	DistributedCache.addCacheFile(modelPath.toUri(), conf);

	Job job = new Job(conf, "Chi_RW classifier");

	log.info("Chi: Configuring the job...");
	configureJob(job);

	log.info("Chi: Running the job...");
	if (!job.waitForCompletion(true)) {
	  throw new IllegalStateException("Chi: Job failed!");
	}

	parseOutput(job);

	HadoopUtil.delete(conf, mappersOutputPath);
  }
  
  /**
   * Extract the prediction for each mapper and write them in the corresponding output file. 
   * The name of the output file is based on the name of the corresponding input file.
   * Will compute the ConfusionMatrix if necessary.
   */
  private void parseOutput(JobContext job) throws IOException {
    Configuration conf = job.getConfiguration();
    FileSystem fs = mappersOutputPath.getFileSystem(conf);

    Path[] outfiles = Chi_RWUtils.listOutputFiles(fs, mappersOutputPath);

    // read all the output
    List<double[]> resList = new ArrayList<double[]>();
    for (Path path : outfiles) {
      FSDataOutputStream ofile = null;
      try {
        for (Pair<DoubleWritable,Text> record : new SequenceFileIterable<DoubleWritable,Text>(path, true, conf)) {
          double key = record.getFirst().get();
          String value = record.getSecond().toString();
          if (ofile == null) {
            // this is the first value, it contains the name of the input file
            ofile = fs.create(new Path(outputPath, value).suffix(".out"));
          } else {
            // The key contains the correct label of the data. The value contains a prediction
            ofile.writeChars(value); // write the prediction
            ofile.writeChar('\n');

            resList.add(new double[]{key, Double.valueOf(value)});
          }
        }
      } finally {
        Closeables.closeQuietly(ofile);
      }
    }
    results = new double[resList.size()][2];
    resList.toArray(results);
  }
  
  /**
   * TextInputFormat that does not split the input files. This ensures that each input file is processed by one single
   * mapper.
   */
  private static class ClassifierTextInputFormat extends TextInputFormat {
    @Override
    protected boolean isSplitable(JobContext jobContext, Path path) {
      return true;
    }
  }
  
  public static class ClassifierMapper extends Mapper<LongWritable, Text, DoubleWritable, Text> {

    /** used to convert input values to data instances */
    private DataConverter converter;
    private final Random rng = RandomUtils.getRandom();
    private boolean first = true;
    private final Text lvalue = new Text();
    private Dataset dataset;
    private final DoubleWritable lkey = new DoubleWritable();
    private RuleBase ruleBase;
    /** will contain all instances if this mapper's split */
    private final List<Instance> instances = Lists.newArrayList();

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      super.setup(context);    //To change body of overridden methods use File | Settings | File Templates.

      Configuration conf = context.getConfiguration();

      URI[] files = DistributedCache.getCacheFiles(conf);

      if (files == null || files.length < 2) {
        throw new IOException("not enough paths in the DistributedCache");
      }
      
      dataset = Dataset.load(conf, new Path(files[0].getPath()));

      converter = new DataConverter(dataset);
      
      ruleBase = RuleBase.load(conf, new Path(files[1].getPath()));  
      
      if (ruleBase == null) {
        throw new InterruptedException("Model not found!");
      }
    }
    /*
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
      instances.add(converter.convert(value.toString()));
    }
    
    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
      for(int i = 0 ; i < instances.size() ; i++){
        double prediction = ruleBase.classify(instances.get(i));
        lkey.set(dataset.getLabel(instances.get(i)));
        lvalue.set(Double.toString(prediction));
        context.write(lkey, lvalue);
      }
    }*/
    
    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
      if (first) {
        FileSplit split = (FileSplit) context.getInputSplit();
        Path path = split.getPath(); // current split path
        lvalue.set(path.getName());
        lkey.set(key.get());
        context.write(lkey, lvalue);

        first = false;
      }

      String line = value.toString();
      if (!line.isEmpty()) {
        Instance instance = converter.convert(line);
        double prediction = ruleBase.classify(instance);
        lkey.set(dataset.getLabel(instance));
        lvalue.set(Double.toString(prediction));
        context.write(lkey, lvalue);
      }
    }
  }

}
