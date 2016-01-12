package se.uu.farmbio.tutorial

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils

object SVM {

  def main(args: Array[String]) = {

    //Start the Spark context
    val conf = new SparkConf()
      .setAppName("SVM")
      .setMaster("local[*]")
    val sc = new SparkContext(conf)

    /*
     * Load the examples in a RDD. The pubchem.svm file is in 
     * LIBSVM format (https://www.csie.ntu.edu.tw/~cjlin/libsvm/),
     * then we can use the loadLibSVMFile function, which is provided 
     * by Spark MLlib, to load it in a RDD. The examples in pubchem.svm 
     * store observations for molecular toxicology, hence the labels, which can be
     * 1 or 0, represent if a molecule is toxic or not. In this dataset 
     * molecular signatures are used as features. 
     */
    val data = MLUtils.loadLibSVMFile(sc, "pubchem.svm")

    /*
     * Split the data. At this point we split the dataset in a training set
     * and test set. The examples in the test set will be used later on
     * in order to evaluate the performance of the obtained model.
     * In particular, 80% of the examples are used as training set, and
     * we leave the remaining 20% for test. 
     */
    val splits = data.randomSplit(Array(0.8, 0.2), seed = 11L) 
    val training = splits(0).cache() //Why do we cache only the training? :-)
    val test = splits(1)

    /*
     * Train the model using linear SVM. Stocastic Gradient Descent 
     * (http://spark.apache.org/docs/latest/mllib-optimization.html#stochastic-gradient-descent-sgd) is used
     * as underlying optimization algorithm here. 
     */
    val numIterations = 100 //Stop SGD after 100 iterations
    val model = SVMWithSGD
      .train(training, numIterations)

    /* 
     * Clear the default threshold. The SVM algorithm computes an hyperplane that 
     * separates positively and negatively labelled examples in the feature spaces. 
     * Then, for a future example the signed distance for this hyperplane
     * can be used in order to predict the unknown label. In fact, if the distance is greater
     * than a certain threshold we classify the new example as positively labeled (toxic), and
     * as negatively labeled (non-toxic) otherwise. Spark uses 0 as default threshold, this means 
     * that if the distance from the hyperplane is greater than 0, then the new example is positively
     * classified, and it is negativaly classified otherwise. However, if we aim to tune sensitivity and
     * specificity of our model, it is good to experiment with different thresholds. In machine learning,
     * the Receiver Operating Characteristic (ROC) curve 
     * (https://en.wikipedia.org/wiki/Receiver_operating_characteristic) is used in order to 
     * study specificity and sensitivity of a certain model, and the area under this curve tells 
     * us how good is our model. In the next line we clear the threshold of the model, so the raw 
     * distance from the separating hyperplane will be output by the model. Doing so we will be 
     * able to plot the ROC curve later on. 
     */
    model.clearThreshold()

    //Compute the distance from the separating hyperplane for each of the test examples
    val scoreAndLabels = test.map { testExample =>
      val distance = model.predict(testExample.features) 
      (distance, testExample.label) 
      /*
       * N.B. we keep track of the label, since we want to compute sensitivity and
       * specificity (in order to plot the ROC curve)
       */
    }

    //Compute the area under the ROC curve using the Spark's BinaryClassificationMetrics class
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC) //print the area under the ROC

    //Stop the Spark context
    sc.stop

  }

}