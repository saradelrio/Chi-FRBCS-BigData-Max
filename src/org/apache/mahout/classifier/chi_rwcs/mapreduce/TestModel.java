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

import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Arrays;

import com.google.common.io.Closeables;
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
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.classifier.chi_rwcs.Chi_RWCSUtils;
import org.apache.mahout.classifier.ResultAnalyzer;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.chi_rwcs.data.Dataset;
import org.apache.mahout.classifier.chi_rwcs.mapreduce.Chi_RWCSClassifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Tool to classify a Dataset using a previously built model
 */
public class TestModel extends Configured implements Tool{
  
  private static final Logger log = LoggerFactory.getLogger(TestModel.class);

  private FileSystem dataFS;
  private Path dataPath; // test data path
  private Path datasetPath;
  private Path modelPath; // path where the model is stored
  private FileSystem outFS;
  private Path outputPath; // path to predictions file, if null do not output the predictions
  private String dataName;
  private long time;
	  
  @Override
  public int run(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
  // TODO Auto-generated method stub
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option inputOpt = DefaultOptionCreator.inputOption().create();

    Option datasetOpt = obuilder.withLongName("dataset").withShortName("ds").withRequired(true).withArgument(
	      abuilder.withName("dataset").withMinimum(1).withMaximum(1).create()).withDescription("Dataset path")
	        .create();

    Option modelOpt = obuilder.withLongName("model").withShortName("m").withRequired(true).withArgument(
	        abuilder.withName("path").withMinimum(1).withMaximum(1).create()).
	        withDescription("Path to the Model").create();

	Option outputOpt = DefaultOptionCreator.outputOption().create();

	Option helpOpt = DefaultOptionCreator.helpOption();

	Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(datasetOpt).withOption(modelOpt)
	        .withOption(outputOpt).withOption(helpOpt).create();

	try {
	  Parser parser = new Parser();
	  parser.setGroup(group);
	  CommandLine cmdLine = parser.parse(args);

	  if (cmdLine.hasOption("help")) {
	    CommandLineUtil.printHelp(group);
	    return -1;
	  }

	  dataName = cmdLine.getValue(inputOpt).toString();
	  String datasetName = cmdLine.getValue(datasetOpt).toString();
	  String modelName = cmdLine.getValue(modelOpt).toString();
	  String outputName = cmdLine.hasOption(outputOpt) ? cmdLine.getValue(outputOpt).toString() : null;
	  
	  if (log.isDebugEnabled()) {
	    log.debug("inout     : {}", dataName);
	    log.debug("dataset   : {}", datasetName);
	    log.debug("model     : {}", modelName);
	    log.debug("output    : {}", outputName);
	  }

	  dataPath = new Path(dataName);
	  datasetPath = new Path(datasetName);
	  modelPath = new Path(modelName);
	  if (outputName != null) {
	    outputPath = new Path(outputName);
	  }
	  
	} catch (OptionException e) {
	  
      log.warn(e.toString(), e);
	  CommandLineUtil.printHelp(group);
	  return -1;
	  
	}
	    
	time = System.currentTimeMillis();
	    
	testModel();
	    
	time = System.currentTimeMillis() - time;
	    
	writeToFileClassifyTime(Chi_RWCSUtils.elapsedTime(time));

    return 0;
  }
  
  private void testModel() throws IOException, ClassNotFoundException, InterruptedException {
	  
	// make sure the output file does not exist
	if (outputPath != null) {
	  outFS = outputPath.getFileSystem(getConf());
	  if (outFS.exists(outputPath)) {
	    throw new IllegalArgumentException("Output path already exists");
	  }
	}
	
	// make sure the model exists
    FileSystem mfs = modelPath.getFileSystem(getConf());
    if (!mfs.exists(modelPath)) {
      throw new IllegalArgumentException("The model path does not exist");
    }
    
    // make sure the test data exists
    dataFS = dataPath.getFileSystem(getConf());
    if (!dataFS.exists(dataPath)) {
      throw new IllegalArgumentException("The Test data path does not exist");
    }
    
    if (outputPath == null) {
      throw new IllegalArgumentException("You must specify the ouputPath when using the mapreduce implementation");
    }
        
    Chi_RWCSClassifier classifier = new Chi_RWCSClassifier(modelPath, dataPath, datasetPath, outputPath, getConf());
    classifier.run();
    
    double[][] results = classifier.getResults();
    if (results != null) {
      writePredictions(results);
      Dataset dataset = Dataset.load(getConf(), datasetPath);      
      ResultAnalyzer analyzer = new ResultAnalyzer(Arrays.asList(dataset.labels()), "unknown");
      for (double[] res : results) {
        analyzer.addInstance(dataset.getLabelString(res[0]), new ClassifierResult(dataset.getLabelString(res[1]), 1.0));
      }          
      parseOutput(analyzer);
    } 
  }
  
  private void parseOutput(ResultAnalyzer analyzer) throws IOException {
    NumberFormat decimalFormatter = new DecimalFormat("0.########");
	outFS = outputPath.getFileSystem(getConf());
	FSDataOutputStream ofile = null;
	int pos=dataName.indexOf('t');
	String subStr=dataName.substring(0, pos);
	Path filenamePath = new Path(outputPath, subStr + "_confusion_matrix").suffix(".txt");
    try    
    {	        	
      if (ofile == null) {
	    // this is the first value, it contains the name of the input file
	    ofile = outFS.create(filenamePath);
		// write the Confusion Matrix	      	      	      	      
		StringBuilder returnString = new StringBuilder(200);	      
		returnString.append("=======================================================").append('\n');
		returnString.append("Confusion Matrix\n");
		returnString.append("-------------------------------------------------------").append('\n');
		int [][] matrix = analyzer.getConfusionMatrix().getConfusionMatrix();	      
		for(int i=0; i< matrix.length-1; i++){
		  for(int j=0; j< matrix[i].length-1; j++){	          	          
		    returnString.append(StringUtils.rightPad(Integer.toString(matrix[i][j]), 5)).append('\t');	
		  } 	        
		  returnString.append('\n');
		}
		returnString.append("-------------------------------------------------------").append('\n');	      	      
		returnString.append("AUC - Area Under the Curve ROC\n");
		returnString.append(StringUtils.rightPad(decimalFormatter.format(computeAuc(matrix)), 5)).append('\n');                  
		returnString.append("-------------------------------------------------------").append('\n');	      
		returnString.append("GM - Geometric Mean\n");
		returnString.append(StringUtils.rightPad(decimalFormatter.format(computeGM(matrix)), 5)).append('\n');                  
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
	 
  private double computeAuc(int [][] matrix){
    int [] classesDistribution = new int [matrix.length-1];  
	for(int i=0; i< matrix.length-1; i++){
      for(int j=0; j< matrix[i].length-1; j++){	          	          
	    classesDistribution[i]+=matrix[i][j];	
	  } 	        	   
	}    
	int posClassId = 0;
	int posNumInstances = classesDistribution[0]; 
	for (int k=1; k<matrix.length-1; k++) {
	  if (classesDistribution[k] < posNumInstances) {
	    posClassId = k;
	 	posNumInstances = classesDistribution[k];
	   }
	}
	double tp_rate = 0.0;
	double fp_rate = 0.0;
	if(posClassId == 0){
	  tp_rate=((double)matrix[0][0]/(matrix[0][0]+matrix[0][1]));	
	  fp_rate=((double)matrix[1][0]/(matrix[1][0]+matrix[1][1]));
	}
	else{
	  fp_rate=((double)matrix[0][1]/(matrix[0][1]+matrix[0][0]));	
	  tp_rate=((double)matrix[1][1]/(matrix[1][1]+matrix[1][0]));	
	}
	return ((1+tp_rate-fp_rate)/2);
  }
	  
  private double computeGM(int [][] matrix){
    int [] classesDistribution = new int [matrix.length-1];  
	for(int i=0; i< matrix.length-1; i++){
	  for(int j=0; j< matrix[i].length-1; j++){	          	          
	    classesDistribution[i]+=matrix[i][j];	
	  } 	        	   
	}    
	int posClassId = 0;
	int posNumInstances = classesDistribution[0]; 
	for (int k=1; k<matrix.length-1; k++) {
	  if (classesDistribution[k] < posNumInstances) {
	    posClassId = k;
		posNumInstances = classesDistribution[k];
	  }
	}
	double sensisivity = 0.0;
	double specificity = 0.0;
	if(posClassId == 0){
	  sensisivity=((double)matrix[0][0]/(matrix[0][0]+matrix[0][1]));	
	  specificity=((double)matrix[1][1]/(matrix[1][1]+matrix[1][0]));
	}
	else{
      specificity=((double)matrix[0][0]/(matrix[0][0]+matrix[0][1]));	
	  sensisivity=((double)matrix[1][1]/(matrix[1][1]+matrix[1][0]));	
    }
	return (Math.sqrt(sensisivity*specificity));  
  }
  
  private void writePredictions(double results[][]) throws IOException {
      NumberFormat decimalFormatter = new DecimalFormat("0.########");
      outFS = outputPath.getFileSystem(getConf());
      FSDataOutputStream ofile = null;
      Path filenamePath = new Path(outputPath, "Predictions").suffix(".txt");
      try   
      {               
        if (ofile == null) {
          // this is the first value, it contains the name of the input file
          ofile = outFS.create(filenamePath);
          // write the Confusion Matrix                                       
          StringBuilder returnString = new StringBuilder();   
         
 
          for (double[] res : results) {
               // returnString.append(res[1]+"\n");
              String dato = Double.toString(res[1])+"\n";
              ofile.writeBytes(dato);

          }
         
          ofile.close();         
        }        
      }
      finally
      {
        Closeables.closeQuietly(ofile);
      }
  }


  
  private void writeToFileClassifyTime(String time) throws IOException{	
    FileSystem outFS = outputPath.getFileSystem(getConf());
	FSDataOutputStream ofile = null;		
	Path filenamePath = new Path(outputPath, dataName + "_classify_time").suffix(".txt");
	try    
	{	        	
      if (ofile == null) {
	    // this is the first value, it contains the name of the input file
	    ofile = outFS.create(filenamePath);
	    // write the Classify Time	      	      	      	      
		StringBuilder returnString = new StringBuilder(200);	      
		returnString.append("=======================================================").append('\n');
		returnString.append("Classify Time\n");
		returnString.append("-------------------------------------------------------").append('\n');
		returnString.append(
			    		  StringUtils.rightPad(time,5)).append('\n');                  
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
    ToolRunner.run(new Configuration(), new TestModel(), args);
  }
}
