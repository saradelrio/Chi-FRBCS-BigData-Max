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
package org.apache.mahout.classifier.chi_rw;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.mahout.classifier.chi_rw.data.Data;
import org.apache.mahout.classifier.chi_rw.data.Dataset;
import org.apache.mahout.classifier.chi_rw.data.Instance;
import org.apache.mahout.classifier.chi_rw.mapreduce.*;

import com.google.common.io.Closeables;

public class RuleBase implements Writable {

    ArrayList<Rule> ruleBase;
    DataBase dataBase;
    int n_variables, n_labels, ruleWeight, inferenceType, compatibilityType;
    String[] names, classes;
    int[] ruleBaseSizes;

    public RuleBase(){
      ruleBase = new ArrayList<Rule>();
    }
    
    /**
     * Rule Base Constructor
     * @param dataBase DataBase the Data Base containing the fuzzy partitions
     * @param inferenceType int the inference type for the FRM
     * @param compatibilityType int the compatibility type for the t-norm
     * @param ruleWeight int the rule weight heuristic
     * @param names String[] the names for the features of the problem
     * @param classes String[] the labels for the class attributes
     */
    public RuleBase(DataBase dataBase, int inferenceType, int compatibilityType, int ruleWeight, String[] names, String[] classes) {
        ruleBase = new ArrayList<Rule>();
        this.dataBase = dataBase;
        n_variables = dataBase.numVariables();
        n_labels = dataBase.numLabels();
        this.inferenceType = inferenceType;
        this.compatibilityType = compatibilityType;
        this.ruleWeight = ruleWeight;
        this.names = names.clone();
        this.classes = classes.clone();
        this.ruleBaseSizes = new int[1];
        Arrays.fill(this.ruleBaseSizes, 0);
    }
    
    /**
     * Rule Base Constructor
     * @param dataBase DataBase the Data Base containing the fuzzy partitions
     * @param inferenceType int the inference type for the FRM
     * @param compatibilityType int the compatibility type for the t-norm
     * @param ruleWeight int the rule weight heuristic
     * @param names String[] the names for the features of the problem
     * @param classes String[] the labels for the class attributes
     */
    public RuleBase(DataBase dataBase, int inferenceType, int compatibilityType, int ruleWeight, String[] names, String[] classes, int mappers) {
        ruleBase = new ArrayList<Rule>();
        this.dataBase = dataBase;
        n_variables = dataBase.numVariables();
        n_labels = dataBase.numLabels();
        this.inferenceType = inferenceType;
        this.compatibilityType = compatibilityType;
        this.ruleWeight = ruleWeight;
        this.names = names.clone();
        this.classes = classes.clone();
        this.ruleBaseSizes = new int[mappers];
    }
    
    public DataBase getDataBase(){
      return dataBase;
    }
    
    public int getInferenceType(){
      return inferenceType;
    }
    
    public int getCompatibilityType(){
      return compatibilityType;
    }
    
    public int getRuleWeight(){
      return ruleWeight;
    }
    
    public String[] getNames(){
      return names;
    }
    
    public String[] getClasses(){
      return classes;
    }
    
    public int size(){
      return ruleBase.size();
    }
    
    public void add(Rule r){
      ruleBase.add(r);	
    }
    
    public Rule get(int i){
      return ruleBase.get(i);
    }
    
    public void setRuleBaseSize(int pos, int size){
      ruleBaseSizes[pos]=size;
    }
    
    public int[] getRuleBaseSizes(){
      return ruleBaseSizes;	
    }
    
    /**
     * It checks if a specific rule is already in the rule base
     * @param r Rule the rule for comparison
     * @return boolean true if the rule is already in the rule base, false in other case
     */
    public boolean duplicated(Rule r) {
        int i = 0;
        boolean found = false;
        while ((i < ruleBase.size()) && (!found)) {
            found = ruleBase.get(i).comparison(r);
            i++;
        }
        return found;
    }

    /**
     * Rule Learning Mechanism for the Chi et al.'s method
     * @param train myDataset the training data-set
     */
    public void Generation(Data train, Context context) {
    	Dataset dataset = train.getDataset();
        for (int i = 0; i < train.size(); i++) {  
        	context.progress();
            Rule r = searchForBestAntecedent(train.get(i).get(), (int) dataset.getLabel(train.get(i)));
            r.assingConsequent(train, ruleWeight);
            if ((!duplicated(r)) &&
                (r.weight > 0)) {
                ruleBase.add(r);
            }
        }
    }

    /**
     * This function obtains the best fuzzy label for each variable of the example and assigns
     * it to the rule
     * @param example double[] the input example
     * @param clas int the class of the input example
     * @return Rule the fuzzy rule with the highest membership degree with the example
     */
    private Rule searchForBestAntecedent(double[] example, int clas) {
        Rule r = new Rule(n_variables, this.compatibilityType);
        r.setClass(clas);
        for (int i = 0; i < n_variables; i++) {
            double max = 0.0;
            int etq = -1;
            double per;
            for (int j = 0; j < n_labels; j++) {
                per = dataBase.membershipFunction(i, j, example[i]);
                if (per > max) {
                    max = per;
                    etq = j;
                }
            }
            if (max == 0.0) {
                System.err.println(
                        "There was an Error while searching for the antecedent of the rule");
                System.err.println("Example: ");
                for (int j = 0; j < n_variables; j++) {
                    System.err.print(example[j] + "\t");
                }
                System.err.println("Variable " + i);
                System.exit(1);
            }
            r.antecedent[i] = dataBase.clone(i, etq);
        }
        return r;
    }

    /**
     * Fuzzy Reasoning Method
     * @param example double[] the input example
     * @return int the predicted class label (id)
     */
    public int FRM(double[] example) {
        if (this.inferenceType == BuildModel.WINNING_RULE) {
            return FRM_WR(example);
        } else {
            return FRM_AC(example);
        }
    }

    /**
     * Winning Rule FRM
     * @param example double[] the input example
     * @return int the class label for the rule with highest membership degree to the example
     */
    private int FRM_WR(double[] example) {
        int clas = -1;
        double max = 0.0;
        for (int i = 0; i < ruleBase.size(); i++) {
            Rule r = ruleBase.get(i);
            double produc = r.compatibility(example);
            produc *= r.weight;
            if (produc > max) {
                max = produc;
                clas = r.clas;
            }
        }
        return clas;
    }

    /**
     * Additive Combination FRM
     * @param example double[] the input example
     * @return int the class label for the set of rules with the highest sum of membership degree per class
     */
    private int FRM_AC(double[] example) {
        int clas = -1;
        double[] class_degrees = new double[1];
        for (int i = 0; i < ruleBase.size(); i++) {
            Rule r = ruleBase.get(i);

            double produc = r.compatibility(example);
            produc *= r.weight;
            if (r.clas > class_degrees.length - 1) {
                double[] aux = new double[class_degrees.length];
                for (int j = 0; j < aux.length; j++) {
                    aux[j] = class_degrees[j];
                }
                class_degrees = new double[r.clas + 1];
                for (int j = 0; j < aux.length; j++) {
                    class_degrees[j] = aux[j];
                }
            }
            class_degrees[r.clas] += produc;
        }
        double max = 0.0;
        for (int l = 0; l < class_degrees.length; l++) {
            if (class_degrees[l] > max) {
                max = class_degrees[l];
                clas = l;
            }
        }
        return clas;
    }
    
    /**
     * It prints the rule base into an string
     * @return String an string containing the rule base
     */
    public String printString() {
        int i, j;
        String cadena = "";

        cadena += "@Number of rules: " + ruleBase.size() + "\n\n";
        for (i = 0; i < ruleBase.size(); i++) {
            Rule r = ruleBase.get(i);
            cadena += (i + 1) + ": ";
            for (j = 0; j < n_variables - 1; j++) {
                cadena += names[j] + " IS " + r.antecedent[j].name + " AND ";
            }
            cadena += names[j] + " IS " + r.antecedent[j].name + ": " +
                    classes[r.clas] + " with Rule Weight: " + r.weight + "\n";
        }

        return (cadena);
    }
    
    /**
     * Load the rule base from a single file or a directory of files
     * @throws java.io.IOException
     */
    public static RuleBase load(Configuration conf, Path fuzzy_ChiCSPath) throws IOException {
      FileSystem fs = fuzzy_ChiCSPath.getFileSystem(conf);
      Path[] files;
      if (fs.getFileStatus(fuzzy_ChiCSPath).isDir()) {
        files = Chi_RWUtils.listOutputFiles(fs, fuzzy_ChiCSPath);
      } else {
        files = new Path[]{fuzzy_ChiCSPath};
      }

      RuleBase rb = null;
      for (Path path : files) {
        FSDataInputStream dataInput = new FSDataInputStream(fs.open(path));
        try {
          if (rb == null) {
            rb = read(dataInput);
          } else {
            rb.readFields(dataInput);
          }
        } finally {
          Closeables.closeQuietly(dataInput);
        }
      }
      return rb;      
    }
    
    private static RuleBase read(DataInput dataInput) throws IOException {
      RuleBase rb = new RuleBase();
      rb.readFields(dataInput);
      return rb;
    }
    
    /**
     * predicts the label for the instance
     * 
     */
    public double classify(Instance instance) {
  	//for classification: 
      return this.classificationOutput(instance.get()); 	
    }
    
    /**
     * It returns the algorithm classification output given an input example
     * @param example double[] The input example
     * @return String the output generated by the algorithm
     */
    private double classificationOutput(double[] example) {
      /**
        Here we should include the algorithm directives to generate the
        classification output from the input example
       */
      int classOut = FRM(example);
      if (classOut >= 0) {
        return classOut;
      }
      else
        return Double.NaN;
    }
    
	@Override
	public void readFields(DataInput in) throws IOException {
		// TODO Auto-generated method stub
		n_variables = in.readInt();
		n_labels = in.readInt();
		ruleWeight = in.readInt();
		inferenceType = in.readInt();
		compatibilityType = in.readInt();
		
		int names_size = in.readInt();
		names = new String[names_size];
		for (int i = 0 ; i < names_size ; i++){
			names[i] = in.readUTF();
		}
		
		int classes_size = in.readInt();
		classes = new String[classes_size];
		for (int i = 0 ; i < classes_size ; i++){
			classes[i] = in.readUTF();
		}
		
		dataBase = new DataBase();
		dataBase.readFields(in);
		
		int ruleBase_size = in.readInt();
		ruleBase = new ArrayList<Rule>();
		for (int i = 0 ; i < ruleBase_size ; i++){
			Rule element = new Rule();
			element.readFields(in);
			ruleBase.add(i, element);
		}		
		
		int ruleBaseSizes_size = in.readInt();
		ruleBaseSizes = new int[ruleBaseSizes_size];
		for (int i = 0 ; i < ruleBaseSizes_size ; i++){
			ruleBaseSizes[i] = in.readInt();
		}
	}
	
	@Override
	public void write(DataOutput out) throws IOException {
		// TODO Auto-generated method stub
		out.writeInt(n_variables);
		out.writeInt(n_labels);
		out.writeInt(ruleWeight);
		out.writeInt(inferenceType);
		out.writeInt(compatibilityType);
		
		out.writeInt(names.length);
		for (int i = 0 ; i < names.length ; i++)
			out.writeUTF(names[i]);			
		
		out.writeInt(classes.length);
		for (int i = 0 ; i < classes.length ; i++)
			out.writeUTF(classes[i]);
		
		dataBase.write(out);
		
		out.writeInt(ruleBase.size());
		for (int i = 0 ; i < ruleBase.size() ; i++)
			ruleBase.get(i).write(out);
		
		out.writeInt(ruleBaseSizes.length);
		for (int i = 0 ; i < ruleBaseSizes.length ; i++)
			out.writeInt(ruleBaseSizes[i]);
	}
}

