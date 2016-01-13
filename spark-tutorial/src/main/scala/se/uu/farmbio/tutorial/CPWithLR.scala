package se.uu.farmbio.tutorial

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils

import se.uu.farmbio.cp.BinaryClassificationICPMetrics
import se.uu.farmbio.cp.ICP
import se.uu.farmbio.cp.alg.LogisticRegression

object CPWithLR {
  
  def main(args: Array[String]) = {

    //Start the Spark context
    val conf = new SparkConf()
      .setAppName("CPWithLR")
      .setMaster("local[*]")
    val sc = new SparkContext(conf)
    
    //Load examples
    val data = MLUtils.loadLibSVMFile(sc, "pubchem.svm")
    
    //Split the data. We leave 20% of the examples out to compute the efficiency later on.
    val splits = data.randomSplit(Array(0.8, 0.2), seed = 11L) 
    val training = splits(0)
    val test = splits(1)
    
    //Split in training and calibration set
    val (calibration, properTraining) =
      ICP.splitCalibrationAndTraining(
          training, 
          //sample 32 calibration examples for each class (balanced calibration set)
          numOfCalibSamples=32, 
          bothClasses=true)
          
    /*
     * Train an inductive conformal predictor in classification settings, using LogisticRegression 
     * as machine learning algorithm.
     */
    val conformalPred = ICP.trainClassifier(
        new LogisticRegression(
            properTraining.cache(), //why do we cache only the proper training set?
            regParam=0.0, //no regularization (for LBFGS)
            maxNumItearations=30), //maximum 30 iterations (for LBFGS)
        numClasses = 2, //we have only two classes: toxic and non-toxic
        calibration) //provide the calibration examples
    
    //Compute the p-values
    val pvAndLabels = test.map { testExample => //for each test example
      val pvForEachClass = conformalPred.mondrianPv(testExample.features) //compute p-value for each class
      val trueLabel = testExample.label //keep track of the true label to compute the efficiency later on
      (pvForEachClass, trueLabel) 
    }
    
    //BinaryClassificationICPMetrics class computes some metrics which include efficiency
    val metrics = new BinaryClassificationICPMetrics(
        pvAndLabels,
        significances=Array(0.1,0.15,0.2,0.25) //specify for which significances the metrics will be computed
    )
    
    //Print the metrics
    println(metrics)
    
    //Stop the Spark context
    sc.stop
    
  }

}