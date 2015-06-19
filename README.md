
#The Chi-FRBCS-BigData algorithm: A MapReduce Design based on the Fusion of Fuzzy Rules

The Chi-FRBCSBigData algorithm uses two different MapReduce processes. One MapReduce process is devoted to the building of the model from a big data training set (Figure 10). The other MapReduce process is used to estimate the class of the examples belonging to big data sample sets using the previous learned model (Figure 11). Both parts follow the MapReduce design, distributing all the computations along several processing units that manage different chunks of information, aggregating the results obtained in an appropriate manner. The two versions developed, which we have named Chi-FRBCS-BigData-Max and Chi-FRBCSBigData-Ave share most of their operations, however, they behave differently in the reduce step of the approach, when the different RBs generated by each map are fused. These versions obtain different RBs and thus, different KBs.
