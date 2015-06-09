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

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.classifier.chi_rwcs.RuleBase;
import org.apache.mahout.classifier.chi_rwcs.mapreduce.MapredOutput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ChiCSReducer extends Reducer<LongWritable, MapredOutput, LongWritable, RuleBase>{
	
	private static final Logger log = LoggerFactory.getLogger(ChiCSReducer.class);
	
	public void reduce(LongWritable key, Iterable<MapredOutput> values, Context context) throws IOException, InterruptedException {
	  LongWritable id = new LongWritable(1);
      RuleBase actualRuleBase;
      RuleBase finalRuleBase = new RuleBase();
    
      for (MapredOutput value : values){  
        actualRuleBase = value.getRuleBase();
	    for(int i = 0 ; i < actualRuleBase.size() ; i++){    	  
    	  if (finalRuleBase.size() == 0){
    	    finalRuleBase = new RuleBase(actualRuleBase.getDataBase(), 
    			  actualRuleBase.getInferenceType(), 
    			  actualRuleBase.getCompatibilityType(), 
    			  actualRuleBase.getRuleWeight(), 
    			  actualRuleBase.getNames(), 
    			  actualRuleBase.getClasses(),
    			  actualRuleBase.getPositive_class(),
    			  actualRuleBase.getPositive_class_cost(),
    			  actualRuleBase.getNegative_class_cost());
    	    finalRuleBase.add(actualRuleBase.get(i));      	  
    	  }else if(!finalRuleBase.duplicated(actualRuleBase.get(i))){
            finalRuleBase.add(actualRuleBase.get(i));
          } 
        }
  	  }
    context.write(id, finalRuleBase);
  }

}
