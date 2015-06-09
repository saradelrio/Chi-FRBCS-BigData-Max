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

import java.io.IOException;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.classifier.chi_rw.Chi_RWUtils;
import org.apache.mahout.classifier.chi_rw.RuleBase;
import org.apache.mahout.classifier.chi_rw.builder.Fuzzy_ChiBuilder;
import org.apache.mahout.classifier.chi_rw.data.Data;
import org.apache.mahout.classifier.chi_rw.data.DataLoader;
import org.apache.mahout.classifier.chi_rw.data.Dataset;
import org.apache.mahout.classifier.chi_rw.mapreduce.partial.PartialBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.io.Closeables;

public class BuildModel extends Configured implements Tool {
  
  private static final Logger log = LoggerFactory.getLogger(BuildModel.class);
  
  private Path dataPath;
  
  private Path datasetPath;

  private Path outputPath;
  
  private Path timePath;
  
  private String dataName;
  
  private String timeName;
  
  private boolean buildTimeIsStored = false;
  
  private long time;
  
  private int nLabels; // Number of labels
  
  int combinationType;
  
  int ruleWeight;
  
  int inferenceType;
  
  public static final int MINIMUM = 0;
  public static final int PRODUCT = 1;
  public static final int CF = 0;
  public static final int PCF_IV = 1;
  public static final int NO_RW = 3;
  public static final int PCF_II = 3;
  public static final int WINNING_RULE = 0;
  public static final int ADDITIVE_COMBINATION = 1;
  

  @Override
  public int run(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
    
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option dataOpt = obuilder.withLongName("data").withShortName("d").withRequired(true)
        .withArgument(abuilder.withName("path").withMinimum(1).withMaximum(1).create())
        .withDescription("Data path").create();
    
    Option datasetOpt = obuilder.withLongName("dataset").withShortName("ds").withRequired(true)
        .withArgument(abuilder.withName("dataset").withMinimum(1).withMaximum(1).create())
        .withDescription("The path of the file descriptor of the dataset").create();
    
    Option timeOpt = obuilder.withLongName("time").withShortName("tm").withRequired(false)
            .withArgument(abuilder.withName("path").withMinimum(1).withMaximum(1).create())
            .withDescription("Time path").create();
    
    Option outputOpt = obuilder.withLongName("output").withShortName("o").withRequired(true)
        .withArgument(abuilder.withName("path").withMinimum(1).withMaximum(1).create())
        .withDescription("Output path, will contain the Decision Forest").create();
    
    Option labelsOpt = obuilder.withLongName("labels").withShortName("l").withRequired(true)
            .withArgument(abuilder.withName("labels").withMinimum(1).withMaximum(1).create())
            .withDescription("Number of Labels").create();
    
    Option combinationTypeOpt = obuilder.withLongName("combinationType").withShortName("t").withRequired(true)
            .withArgument(abuilder.withName("combinationType").withMinimum(1).withMaximum(1).create())
            .withDescription("T-norm for the computation of the compatibility degree").create();

    Option rule_weightOpt = obuilder.withLongName("rule_weight").withShortName("r").withRequired(true)
            .withArgument(abuilder.withName("rule_weight").withMinimum(1).withMaximum(1).create())
            .withDescription("Rule Weight").create();
    
    Option fuzzy_r_mOpt = obuilder.withLongName("fuzzy_r_m").withShortName("f").withRequired(true)
            .withArgument(abuilder.withName("fuzzy_r_m").withMinimum(1).withMaximum(1).create())
            .withDescription("Fuzzy Reasoning Method").create();
    
    Option helpOpt = obuilder.withLongName("help").withShortName("h")
        .withDescription("Print out help").create();
    
    Group group = gbuilder.withName("Options").withOption(dataOpt).withOption(datasetOpt).withOption(timeOpt)
    		.withOption(outputOpt).withOption(labelsOpt).withOption(combinationTypeOpt).withOption(rule_weightOpt).withOption(fuzzy_r_mOpt)
    		.withOption(helpOpt).create();
    
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      
      if (cmdLine.hasOption("help")) {
        CommandLineUtil.printHelp(group);
        return -1;
      }
    
      dataName = cmdLine.getValue(dataOpt).toString();
      String datasetName = cmdLine.getValue(datasetOpt).toString();
      String outputName = cmdLine.getValue(outputOpt).toString();
      nLabels = Integer.parseInt(cmdLine.getValue(labelsOpt).toString());      
      String combinationType_aux = cmdLine.getValue(combinationTypeOpt).toString();
      String ruleWeight_aux = cmdLine.getValue(rule_weightOpt).toString();
      String inferenceType_aux = cmdLine.getValue(fuzzy_r_mOpt).toString();      
      
      if (cmdLine.hasOption(timeOpt)) {
      	buildTimeIsStored = true;  
        timeName = cmdLine.getValue(timeOpt).toString();
      } 

      if (log.isDebugEnabled()) {
        log.debug("data : {}", dataName);
        log.debug("dataset : {}", datasetName);
        log.debug("output : {}", outputName);
        log.debug("labels : {}", nLabels);
        log.debug("t_norm : {}", combinationType_aux);
        log.debug("rule_weight : {}", ruleWeight_aux);
        log.debug("fuzzy_r_m : {}", inferenceType_aux);
        log.debug("time : {}", timeName);
      }

      dataPath = new Path(dataName);
      datasetPath = new Path(datasetName);
      outputPath = new Path(outputName);
      if(buildTimeIsStored)
          timePath = new Path(timeName);
      
      combinationType = PRODUCT;
      if (combinationType_aux.compareToIgnoreCase("minimum") == 0) {
        combinationType = MINIMUM;
      }
      
      ruleWeight = PCF_IV;
      if (ruleWeight_aux.compareToIgnoreCase("Certainty_Factor") == 0) {
        ruleWeight = CF;
      }
      else if (ruleWeight_aux.compareToIgnoreCase("Average_Penalized_Certainty_Factor") == 0) {
        ruleWeight = PCF_II;
      }
      else if (ruleWeight_aux.compareToIgnoreCase("No_Weights") == 0){
        ruleWeight = NO_RW;
      }
      
      inferenceType = WINNING_RULE;
      if (inferenceType_aux.compareToIgnoreCase("Additive_Combination") == 0) {
        inferenceType = ADDITIVE_COMBINATION;
      }
      
    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
      return -1;
    }
    
    buildModel();
    
    return 0;
  }
  
  private void buildModel() throws IOException, ClassNotFoundException, InterruptedException {
    // make sure the output path does not exist
    FileSystem ofs = outputPath.getFileSystem(getConf());
    if (ofs.exists(outputPath)) {
      log.error("Output path already exists");
      return;
    }

    Fuzzy_ChiBuilder fuzzy_ChiBuilder = new Fuzzy_ChiBuilder();
    
    fuzzy_ChiBuilder.setCombinationType(combinationType);
      
    fuzzy_ChiBuilder.setInferenceType(inferenceType);
    
    fuzzy_ChiBuilder.setNLabels(nLabels);
    
    fuzzy_ChiBuilder.setRuleWeight(ruleWeight);
    
    Builder modelBuilder;

    log.info("Chi: Partial Mapred implementation");
        
    modelBuilder = new PartialBuilder(fuzzy_ChiBuilder, dataPath, datasetPath, getConf());       

    modelBuilder.setOutputDirName(outputPath.getName());
    
    log.info("Chi: Building the model...");
    
    time = System.currentTimeMillis();
    
    RuleBase ruleBase = modelBuilder.build();
    
    time = System.currentTimeMillis() - time;
    
    int ruleBaseSize = ruleBase.size();
    
    int [] ruleBasesSize = ruleBase.getRuleBaseSizes();
    
    if(buildTimeIsStored)
        writeToFileBuildTime(Chi_RWUtils.elapsedTime(time), ruleBaseSize, ruleBasesSize);
    
    log.info("Chi: Build Time: {}", Chi_RWUtils.elapsedTime(time));
    
    // store the model in the output path
    Path modelPath = new Path(outputPath, "model.seq");
    log.info("ChiCS: Storing the model in: {}", modelPath);
    Chi_RWUtils.storeWritable(getConf(), modelPath, ruleBase);
  }
  
  protected static Data loadData(Configuration conf, Path dataPath, Dataset dataset) throws IOException {
    log.info("Chi: Loading the data...");
    FileSystem fs = dataPath.getFileSystem(conf);
    Data data = DataLoader.loadData(dataset, fs, dataPath);
    log.info("Chi: Data Loaded");
    
    return data;
  }
  
  private void writeToFileBuildTime(String time, int ruleBaseSize, int [] ruleBasesSize) throws IOException{	
    FileSystem outFS = timePath.getFileSystem(getConf());
	FSDataOutputStream ofile = null;		
	Path filenamePath = new Path(timePath, dataName + "_build_time").suffix(".txt");
	try    
	  {	        	
        if (ofile == null) {
	      // this is the first value, it contains the name of the input file
	      ofile = outFS.create(filenamePath);
		  // write the Build Time	      	      	      	      
		  StringBuilder returnString = new StringBuilder(200);	      
	      returnString.append("=======================================================").append('\n');
		  returnString.append("Build Time\n");
		  returnString.append("-------------------------------------------------------").append('\n');
		  returnString.append(
		    		  StringUtils.rightPad(time,5)).append('\n');                  
		  returnString.append("-------------------------------------------------------").append('\n');	
		  returnString.append("Rule Base Size\n");
		  returnString.append("-------------------------------------------------------").append('\n');
		  returnString.append(
		    		  StringUtils.rightPad(Integer.toString(ruleBaseSize),5)).append('\n');    		  
		  returnString.append("-------------------------------------------------------").append('\n');	
		  returnString.append("Rule Bases Size\n");
		  returnString.append("-------------------------------------------------------").append('\n');
		  for (int i = 0 ; i < ruleBasesSize.length ; i++){
		    returnString.append(
		    		  StringUtils.rightPad(Integer.toString(ruleBasesSize[i]),5)).append('\n');   
		  }
		  returnString.append("-------------------------------------------------------").append('\n');
		  String output = returnString.toString();
	      ofile.writeUTF(output);
		  ofile.close();		  
		} 	    
      } 
	finally 
      {
	    Closeables.closeQuietly(ofile);
	  }
  }
  
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new BuildModel(), args);
  }
  
}