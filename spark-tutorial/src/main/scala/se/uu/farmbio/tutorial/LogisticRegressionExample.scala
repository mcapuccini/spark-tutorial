package se.uu.farmbio.tutorial

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils

object LogisticRegressionExample {

  def main(args: Array[String]) = {

    //Start the Spark context
    val conf = new SparkConf()
      .setAppName("LogisticRegression")
      .setMaster("local[*]")
    val sc = new SparkContext(conf)

    //Load pubchem.svm
    val data = MLUtils.loadLibSVMFile(sc, "pubchem.svm")

    //Split the data in training and test
    val splits = data.randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    //Train the model using Logistic Regression with LBFGS
    val lbfgs = new LogisticRegressionWithLBFGS()
    val model = lbfgs.run(training)

    /*
     * Clear the threshold. Logistic Regression trains a model that outputs
     * the probability for a new example to be in the positive class. This can 
     * be used with a threshold in order to predict the unknown label of a new example.
     * Like we did in the SVM example, we clear the threshold, hence the raw probability
     * will be output by the model. This will allow us to plot the ROC curve. 
     */
    model.clearThreshold()

    //Compute the probability to be in the positive class for each of the test examples
    val probAndLabels = test.map { testExample =>
      val probability = model.predict(testExample.features)
      (probability, testExample.label)
    }

    //Compute the area under the ROC curve using the Spark's BinaryClassificationMetrics class
    val metrics = new BinaryClassificationMetrics(probAndLabels)
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC) //print the area under the ROC

    //Stop the Spark context
    sc.stop

  }

}
