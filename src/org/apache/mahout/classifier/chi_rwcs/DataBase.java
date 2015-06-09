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

public class DataBase implements Writable{
    int n_variables;
    int n_labels;
    Fuzzy[][] dataBase;
    String[] names;

    /**
     * Default constructor
     */
    public DataBase() {
    }

    /**
     * Constructor with parameters. It performs a homegeneous partition of the input space for
     * a given number of fuzzy labels.
     * @param n_variables int Number of input variables of the problem
     * @param n_labels int Number of fuzzy labels
     * @param rangos double[][] Range of each variable (minimum and maximum values)
     * @param names String[] Labels for the input attributes
     */
    public DataBase(int n_variables, int n_labels, double[][] rangos, String[] names) {
        this.n_variables = n_variables;
        this.n_labels = n_labels;
        dataBase = new Fuzzy[n_variables][n_labels];
        this.names = names.clone();

        double marca;
        for (int i = 0; i < n_variables; i++) {
            marca = (rangos[i][1] - rangos[i][0]) / ((double) n_labels - 1);
            if (marca == 0) { //there are no ranges (an unique valor)
                for (int etq = 0; etq < n_labels; etq++) {
                    dataBase[i][etq] = new Fuzzy();
                    dataBase[i][etq].x0 = rangos[i][1] - 0.00000000000001;
                    dataBase[i][etq].x1 = rangos[i][1];
                    dataBase[i][etq].x3 = rangos[i][1] + 0.00000000000001;
                    dataBase[i][etq].y = 1;
                    dataBase[i][etq].name = new String("L_" + etq);
                    dataBase[i][etq].label = etq;
                }
            } else {
                for (int etq = 0; etq < n_labels; etq++) {
                    dataBase[i][etq] = new Fuzzy();
                    dataBase[i][etq].x0 = rangos[i][0] + marca * (etq - 1);
                    dataBase[i][etq].x1 = rangos[i][0] + marca * etq;
                    dataBase[i][etq].x3 = rangos[i][0] + marca * (etq + 1);
                    dataBase[i][etq].y = 1;
                    dataBase[i][etq].name = new String("L_" + etq);
                    dataBase[i][etq].label = etq;
                }
            }
        }
    }

    /**
     * It returns the number of input variables
     * @return int the number of input variables
     */
    public int numVariables() {
        return n_variables;
    }

    /**
     * It returns the number of fuzzy labels
     * @return int the number of fuzzy labels
     */
    public int numLabels() {
        return n_labels;
    }

    /**
     * It computes the membership degree for a input value
     * @param i int the input variable id
     * @param j int the fuzzy label id
     * @param X double the input value
     * @return double the membership degree
     */
    public double membershipFunction(int i, int j, double X) {
        return dataBase[i][j].Fuzzify(X);
    }

    /**
     * It makes a copy of a fuzzy label
     * @param i int the input variable id
     * @param j int the fuzzy label id
     * @return Fuzzy a copy of a fuzzy label
     */
    public Fuzzy clone(int i, int j) {
        return dataBase[i][j].clone();
    }

    /**
     * It prints the Data Base into an string
     * @return String the data base
     */
    public String printString() {
        String cadena = new String(
                "@Using Triangular Membership Functions as antecedent fuzzy sets\n");
        cadena += "@Number of Labels per variable: " + n_labels + "\n";
        for (int i = 0; i < n_variables; i++) {
            //cadena += "\nVariable " + (i + 1) + ":\n";
            cadena += "\n" + names[i] + ":\n";
            for (int j = 0; j < n_labels; j++) {
                cadena += " L_" + (j + 1) + ": (" + dataBase[i][j].x0 +
                        "," + dataBase[i][j].x1 + "," + dataBase[i][j].x3 +
                        ")\n";
            }
        }
        return cadena;
    }

   
	@Override
	public void readFields(DataInput in) throws IOException {
		// TODO Auto-generated method stub
		n_variables = in.readInt();
		n_labels = in.readInt();
		
		int names_size = in.readInt();
		names = new String[names_size];
		for (int i = 0 ; i < names_size ; i++){
			names[i] = in.readUTF();
		}
		
		dataBase = new Fuzzy[n_variables][n_labels];
		for(int f=0;f<n_variables;f++) {
            for(int c=0;c<n_labels;c++) {               	
            	dataBase[f][c] = new Fuzzy();
            	dataBase[f][c].readFields(in);
            }
        }	
	}

	@Override
	public void write(DataOutput out) throws IOException {
		// TODO Auto-generated method stub
		out.writeInt(n_variables);
		out.writeInt(n_labels);
		
		out.writeInt(names.length);
		for (int i = 0 ; i < names.length ; i++)
			out.writeUTF(names[i]);
		
		for(int f = 0 ; f < n_variables; f++) {
            for(int c = 0 ; c < n_labels ; c++) {                
            	dataBase[f][c].write(out);
            }
        }			
	}
}

