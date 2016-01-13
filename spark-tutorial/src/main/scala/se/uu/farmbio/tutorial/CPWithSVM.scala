package se.uu.farmbio.tutorial

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils
import se.uu.farmbio.cp.ICP
import se.uu.farmbio.cp.alg.SVM
import se.uu.farmbio.cp.BinaryClassificationICPMetrics

object CPWithSVM {
  
  def main(args: Array[String]) = {

    //Start the Spark context
    val conf = new SparkConf()
      .setAppName("CPWithSVM")
      .setMaster("local[*]")
    val sc = new SparkContext(conf)
    
    //Load examples
    val data = MLUtils.loadLibSVMFile(sc, "pubchem.svm")
    
    //Split the data. We leave 20% of the examples out to compute the efficiency later on.
    val splits = data.randomSplit(Array(0.8, 0.2), seed = 11L) 
    val training = splits(0)
    val test = splits(1)
    
    /*
     * Split the training set in a calibration set and a proper training set. We implemented 
     * conformal prediction in Spark using an inductive approach (http://cdn.intechweb.org/pdfs/5294.pdf),
     * which requires to provide a small set of calibration examples, which needs to be disjoint from
     * the examples that will be used to train the machine learning model. 
     */
    val (calibration, properTraining) =
      ICP.splitCalibrationAndTraining(
          training, 
          //sample 32 calibration examples for each class (balanced calibration set)
          numOfCalibSamples=32, 
          bothClasses=true)
          
    /*
     * Train an inductive conformal predictor in classification settings, using SVM 
     * as machine learning algorithm.
     */
    val conformalPred = ICP.trainClassifier(
        new SVM(
            properTraining.cache(), //why do we cache only the proper training set?
            regParam=0.0, //no regularization (for LBFGS)
            maxNumItearations=30), //maximum 30 iterations (for LBFGS)
        numClasses = 2, //we have only two classes: toxic and non-toxic
        calibration) //provide the calibration examples
    
    /*
     * Compute p-values for each of the test examples. Instead of computing the prediction sets for a 
     * certain significance level, we use the conformal predictior in order to compute the p-value 
     * with respect to each possible label. The p-value is used by the conformal predictor, along with the
     * user-defined significance level (which acts as a threshold), in order to asses if a label 
     * should be included in the prediction set. In we compute the p-values at this point we will
     * be able to compute the efficiency for multiple significance levels later on. 
     */
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