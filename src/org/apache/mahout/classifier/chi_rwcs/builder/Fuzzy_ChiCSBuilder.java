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
package org.apache.mahout.classifier.chi_rwcs.builder;

import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.mahout.classifier.chi_rwcs.data.Data;
import org.apache.mahout.classifier.chi_rwcs.data.Dataset;
import org.apache.mahout.classifier.chi_rwcs.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class Fuzzy_ChiCSBuilder {
  
  private static final Logger log = LoggerFactory.getLogger(Fuzzy_ChiCSBuilder.class);	
  int nClasses, nLabels, combinationType, inferenceType, ruleWeight;
  DataBase dataBase;
  RuleBase ruleBase;
  
  public void setNLabels(int nLabels) {
    this.nLabels = nLabels;
  }

  public void setCombinationType(int combinationType) {
    this.combinationType = combinationType;
  }	
  
  public void setInferenceType(int inferenceType) {
    this.inferenceType = inferenceType;
  }	
  
  public void setRuleWeight(int ruleWeight) {
    this.ruleWeight = ruleWeight;
  }	
  
  public DataBase getDataBase() {
    return this.dataBase;
  }	
  
  public RuleBase getRuleBase() {
    return this.ruleBase;
  }	

  public void build(Data data, int positive_class, double positive_class_cost, double negative_class_cost, Context context) {
    //We do here the algorithm's operations

	Dataset dataset = data.getDataset();
	 
	nClasses = dataset.nblabels();
	
	//Gets the number of input attributes of the data-set
	int nInputs = dataset.nbAttributes() - 1;
	
	//It returns the class labels
	String clases[] = dataset.labels();
	
	dataBase = new DataBase(nInputs, nLabels, data.getDataset().getRanges(), data.getNames());
	
	System.out.println(dataBase.printString());
	
	ruleBase = new RuleBase(dataBase, inferenceType, combinationType, ruleWeight, data.getNames(), clases, positive_class, positive_class_cost, negative_class_cost);	
	
	ruleBase.Generation(data, context);
	
	System.out.println(ruleBase.printString());
  }


}
