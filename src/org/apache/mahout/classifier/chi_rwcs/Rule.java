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
package org.apache.mahout.classifier.chi_rwcs;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.chi_rwcs.data.Data;
import org.apache.mahout.classifier.chi_rwcs.data.Dataset;
import org.apache.mahout.classifier.chi_rwcs.mapreduce.*;

public class Rule implements Writable{

	  Fuzzy[] antecedent;
	  int clas;
	  double weight;
	  int compatibilityType;

	  /**
	   * Default constructor
	   */
	  public Rule() {
	  }

	  /**
	   * Constructor with parameters
	   * @param n_variables int
	   * @param compatibilityType int
	   */
	  public Rule(int n_variables, int compatibilityType) {
	    antecedent = new Fuzzy[n_variables];
	    this.compatibilityType = compatibilityType;
	  }

	  /**
	   * It assigns the class of the rule
	   * @param clas int
	   */
	  public void setClass(int clas) {
	      this.clas = clas;
	  }

	  /**
	   * It assigns the rule weight to the rule
	   * @param train myDataset the training set
	   * @param ruleWeight int the type of rule weight
	   */
	  public void assingConsequent(Data train, int ruleWeight, int positive_class, double positive_cost, double negative_cost) {
	    if (ruleWeight == BuildModel.CF) {
	      consequent_CF (train, positive_class, positive_cost, negative_cost);
	    }
	    else if (ruleWeight == BuildModel.PCF_II) {
	      consequent_PCF2(train, positive_class, positive_cost, negative_cost);
	    }
	    else if (ruleWeight == BuildModel.PCF_IV) {
	      consequent_PCF4(train, positive_class, positive_cost, negative_cost);
	    }
	    else if (ruleWeight == BuildModel.NO_RW) {
	      weight = 1.0;
	    }
	  }

	  /**
	   * It computes the compatibility of the rule with an input example
	   * @param example double[] The input example
	   * @return double the degree of compatibility
	   */
	  public double compatibility(double[] example) {
	    if (compatibilityType == BuildModel.MINIMUM) {
	      return minimumCompatibility(example);
	    }
	    else {
	      return productCompatibility(example);
	    }
	  }

	  /**
	   * Operator T-min
	   * @param example double[] The input example
	   * @return double the computation the the minimum T-norm
	   */
	  private double minimumCompatibility(double[] example) {
	    double minimum, membershipDegree;
	    minimum = 1.0;
	    for (int i = 0; i < antecedent.length; i++) {
	      membershipDegree = antecedent[i].Fuzzify(example[i]);
	      minimum = Math.min(membershipDegree, minimum);
	    }
	    return (minimum);

	  }

	  /**
	   * Operator T-product
	   * @param example double[] The input example
	   * @return double the computation the the product T-norm
	   */
	  private double productCompatibility(double[] example) {
	    double product, membershipDegree;
	    product = 1.0;
	    for (int i = 0; i < antecedent.length; i++) {
	      membershipDegree = antecedent[i].Fuzzify(example[i]);
	      product = product * membershipDegree;
	    }
	    return (product);
	  }

	  /**
	   * Classic Certainty Factor weight
	   * @param train myDataset training dataset
	   */
	  private void consequent_CF (Data train, int positive_class, double positive_cost, double negative_cost) {		  
		Dataset dataset = train.getDataset();	               
	    double[] classes_sum = new double[dataset.nblabels()];
	    for (int i = 0; i < dataset.nblabels(); i++) {
	      classes_sum[i] = 0.0;
	    }

	    double total = 0.0;
	    double comp;
	    /* Computation of the sum by classes */
	    for (int i = 0; i < train.size(); i++) {
	      comp = this.compatibility(train.get(i).get());
	      if ((int) dataset.getLabel(train.get(i)) == positive_class) {
	          classes_sum[(int) dataset.getLabel(train.get(i))] = classes_sum[(int) dataset.getLabel(train.get(i))] + (positive_cost * comp);
	          total = total + (positive_cost * comp);
	      }
	      else {
	          classes_sum[(int) dataset.getLabel(train.get(i))] = classes_sum[(int) dataset.getLabel(train.get(i))] + (negative_cost * comp);
	          total = total + (negative_cost * comp);
	      }
	    }
	    weight = classes_sum[clas] / total;
	  }

	  /**
	   * Penalized Certainty Factor weight II (by Ishibuchi)
	   * @param train myDataset training dataset
	   */
	  private void consequent_PCF2(Data train, int positive_class, double positive_cost, double negative_cost) {
		Dataset dataset = train.getDataset();	               
		double[] classes_sum = new double[dataset.nblabels()];
		for (int i = 0; i < dataset.nblabels(); i++) {
		  classes_sum[i] = 0.0;
		}

	    double total = 0.0;
	    double comp;
	    /* Computation of the sum by classes */
	    for (int i = 0; i < train.size(); i++) {
	      comp = this.compatibility(train.get(i).get());
	      if ((int) dataset.getLabel(train.get(i)) == positive_class) {
	          classes_sum[(int) dataset.getLabel(train.get(i))] = classes_sum[(int) dataset.getLabel(train.get(i))] + (positive_cost * comp);
	          total = total + (positive_cost * comp);
	      }
	      else {
	          classes_sum[(int) dataset.getLabel(train.get(i))] = classes_sum[(int) dataset.getLabel(train.get(i))] + (negative_cost * comp);
	          total = total + (negative_cost * comp);
	      }
	    }
	    double sum = (total - classes_sum[clas]) / (dataset.nblabels() - 1.0);
	    weight = (classes_sum[clas] - sum) / total;
	  }

	  /**
	   * Penalized Certainty Factor weight IV (by Ishibuchi)
	   * @param train myDataset training dataset
	   */
	  private void consequent_PCF4(Data train, int positive_class, double positive_cost, double negative_cost) {
		Dataset dataset = train.getDataset();	               
		double[] classes_sum = new double[dataset.nblabels()];
		for (int i = 0; i < dataset.nblabels(); i++) {
		  classes_sum[i] = 0.0;
		}

	    double total = 0.0;
	    double comp;
	    /* Computation of the sum by classes */
	    for (int i = 0; i < train.size(); i++) {
	      comp = this.compatibility(train.get(i).get());
	      if ((int) dataset.getLabel(train.get(i)) == positive_class) {
	          classes_sum[(int) dataset.getLabel(train.get(i))] = classes_sum[(int) dataset.getLabel(train.get(i))] + (positive_cost * comp);
	          total = total + (positive_cost * comp);
	      }
	      else {
	          classes_sum[(int) dataset.getLabel(train.get(i))] = classes_sum[(int) dataset.getLabel(train.get(i))] + (negative_cost * comp);
	          total = total + (negative_cost * comp);
	      }
	    }
	    double sum = total - classes_sum[clas];
	    weight = (classes_sum[clas] - sum) / total;
	  }

	  /**
	   * This function detects if one rule is already included in the Rule Set
	   * @param r Rule Rule to compare
	   * @return boolean true if the rule already exists, else false
	   */
	  public boolean comparison(Rule r) {
	    for (int j = 0; j < antecedent.length; j++) {
	      if (this.antecedent[j].label != r.antecedent[j].label) {
	        return false;
	      }
	    }
	    if (this.clas != r.clas) { //Comparison of the rule weights
	      if (this.weight < r.weight) {
	        //Rule Update
	        this.clas = r.clas;
	        this.weight = r.weight;
	      }
	    }
	    return true;
	  }

	@Override
	public void readFields(DataInput in) throws IOException {
		// TODO Auto-generated method stub
		int antecedent_size = in.readInt();
		antecedent = new Fuzzy[antecedent_size];
		for (int i = 0 ; i < antecedent.length ; i++){
			antecedent[i] = new Fuzzy();
			antecedent[i].readFields(in);
		}
		
		clas = in.readInt();
		weight = in.readDouble();
		compatibilityType = in.readInt();		
	}

	@Override
	public void write(DataOutput out) throws IOException {
		// TODO Auto-generated method stub
		out.writeInt(antecedent.length);
		for (int i = 0 ; i < antecedent.length ; i++)
			antecedent[i].write(out);
		
		out.writeInt(clas);
		out.writeDouble(weight);
		out.writeInt(compatibilityType);		
	}
}

