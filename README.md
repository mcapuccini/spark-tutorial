# Introduction to predictive modeling in Spark with applications in pharmaceutical bioinformatics
[Spark](http://spark.apache.org/) is a genearal cluster computing engine for large-scale data processing. In this repository we store tutorials, exercises, and code snippets, to get started with Spark for predictive modeling. Even if the main focus here is pharmaceutical bioinformatics, the presented methodologies are generally applicable, hence the following tutorials represent a good starting point for everyone is interested in learning Spark. 

In this repository we code Spark programs in Scala, if you are not familiar with it please first give a look to this [tutorial](https://scalatutorials.com/tour/). 

## Getting started
In order to get your hands dirty, writing and testing some Spark-based code, you will first need to access a Spark installation. Off course, cloud resources and Spark bare-metal installations are expensive options, both in terms of money and set up time. Therefore, to get started it is preferable to setup a single machine environment on your computer. This will be useful even later on, when you will deploy your Spark applications to a production environment, as you always want to test your code locally first. 

Please follow this video tutorial to setup Spark on your local machine: https://www.youtube.com/watch?v=aB4-RD_MMf0. 

### Scala IDE
If you followed the video tutoria,l you probably noticed that my recipe to get Spark on your local machine is a bit unconventional. I believe that IDEs improve software developers productivity, therefore **Scala IDE** is the main ingredient (http://scala-ide.org/). 

Scala IDE comes with an integrated **Maven** plugin, that can be used to pull Spark and all of its dependencies. Furthermore, Maven can automatically build your Scala code into production-ready jar packages. All you need to do is to configure your *pom.xml* file properly.

Another important remark is that, using Scala IDE, it is simple to sync your code to **GitHub**. 

### Word count
The **word count** problem is considered to be the *"hello world"* of big data analytics. The task performed by a word count program is very simple: *given a text file, count how many times every word occurs*. 

```scala
package se.uu.farmbio.tutorial

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object WordCount {
  
  def main(args: Array[String]) = {
    
    //Start the Spark context
    val conf = new SparkConf()
      .setAppName("WordCount")
      .setMaster("local")
    val sc = new SparkContext(conf)
    
    //Read some example file to a test RDD
    val test = sc.textFile("food.txt")
    
    test.flatMap { line => //for each line
      line.split(" ") //split the line in word by word. NB: we use flatMap because we return a list
    }
    .map { word => //for each word
      (word,1) //return a key/value tuple, with the word as key and 1 as value
    }
    .reduceByKey(_ + _) //sum all of the value with same key
    .saveAsTextFile("food.counts.txt") //save to a text file
    
    //Stop the Spark context
    sc.stop
    
  }

}
```
Please watch the getting started video tutorial for further details on the previous code snippet.

##Exercise 1: Consensus sequence in Spark
Computing the **consensus** for a sequence alignment is an interesting problem in bioinfarmatics (https://en.wikipedia.org/wiki/Consensus_sequence). Given an alignment of several sequences, we want to find the consensus sequence, that is *the sequence that has the most frequent residue in each position of the alignment*.

**Example**  
```no-highlight
Sequence 1: A C C T G  
Sequence 2: G G G T C  
Sequence 3: A C A T C  
Sequence 4: C C G G C  
---------------------
Consensus : A C G T C
```
 **Task:** write a Spark program that, given a text file containing a sequence alignment (where each sequence is stored in a separate line), computes the consensus sequence. For simplicity you can assume that there is no gap in the alignment, and that each sequence has the same length. You can use this [example file](https://github.com/mcapuccini/spark-tutorial/blob/master/spark-tutorial/dna.txt) as input in order to test your code. 

**Hints:**

 1. In the *word count* example we group by word, as we are in interested in counting word-wise. In the consensus problem we are interested in finding the most frequent residue at each position, so first we need to *group each residue position-wise*. 
 2.  Once each residue is nicely grouped by position, we aim to find the most frequent one, in each of the groups. Hence we just need to map every group to the most frequent residue in it. 
 3. Remember that we need the consensus to be sorted position-wise. Therefore, don't forget to keep track of the position of each most frequent residue, so you will be able to use *sortByKey* (RDD transform) in order to sort your result. 

**Solution:** you can give a look to the solution [here](https://github.com/mcapuccini/spark-tutorial/blob/master/spark-tutorial/src/main/scala/se/uu/farmbio/tutorial/Consensus.scala), but first try it yourself :smirk:

##Predictive modelling in Spark
In **predictive modelling**, basing on previous observations, we aim to build a statistical model to predict the future behaviour of a certain system. A *predictive model* is a function to predict future outcomes, basing on a number of *features*, which provide the means to describe an event in the system that we aim to model. In *pharmaceutical bioinformatics*,  predictive modelling is used in order to predict molecular behaviours, such as *binding affinity* or *toxicology*. [Molecular descriptors](https://en.wikipedia.org/wiki/Molecular_descriptor) such as [log P](https://en.wikipedia.org/wiki/Log_P), [molar refractivity](https://en.wikipedia.org/wiki/Molar_refractivity), [dipole moment](https://en.wikipedia.org/wiki/Molecular_dipole_moment), [polarizability](https://en.wikipedia.org/wiki/Polarizability), and [molecular signatures](http://www.ncbi.nlm.nih.gov/pubmed/15032522),  are mostly used as features to make up such predictive models. 

In predictive modelling, we call **training examples** the previous observations that we use in order to *train* the predictive model. Each of the training examples stores a system outcome, that we call *label*, and a features vector that describes that outcome. Of course, choosing the right set of features to describe the behaviour that we are trying to model is crucial. An example follows. 

**Training examples**  

| Toxicity (label)  | Log P (feature 1) | Polarizability (feature 2) | ... |  Dipole moment (fearure n) |
| -------------  | ----------------- | -------------------------- | ----| ----  |
| 1.0 (yes)      | 0.450...          | 1.309...                   | ... | 1.297 |
| 0.0 (no)       | 0.346...          | 3.401...                   | ... | 0.281 |
| 1.0 (yes)      | 4.446...          | 2.431...                   | ... | 6.741 |
| 0.0 (no)       | 3.306...          | 0.473...                   | ... | 1.365 |
| ... | ... | ... | ... | ... |


Basing on the training examples, a variety of **machine leaning** algorithms can be used in order to train a model. [Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine) (SVM) have been successfully used in order to build predictive models in pharmaceutical bioinformatics.  A linear SVM implementation is provided by the [Spark MLlib](http://spark.apache.org/mllib/) package. This enables predictive modelling for pharmaceutical bioinformatics over big training datasets. The following code snippet shows how to train a predictive model using the Spark SVM implementation. 
 
```scala
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
     * Train the model using linear SVM. Stochastic Gradient Descent 
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
```

**Task:** try to run the previous code snippet on your machine, using [pubchem.svm](https://raw.githubusercontent.com/mcapuccini/spark-tutorial/master/spark-tutorial/pubchem.svm) as input. How good is the area under the ROC curve?

###SVM with LBFGS optimization
In the previous code snippet we trained the model using SVM, with the default [Stocastic Gradient Descent](http://spark.apache.org/docs/latest/mllib-optimization.html#stochastic-gradient-descent-sgd) (SGD) algorithm. This happens to work poorly with molecular signatures, maybe because it is designed to deal with really huge data (e.g. streams of tweets). However, Spark provides [LBFGS](http://spark.apache.org/docs/latest/mllib-optimization.html#l-bfgs) as an alternative to SGD.  Hence, the previous code can be adapted to use LBFGS in order to improve the model performance. 

```scala
package se.uu.farmbio.tutorial

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.HingeGradient
import org.apache.spark.mllib.optimization.LBFGS
import org.apache.spark.mllib.optimization.SimpleUpdater
import org.apache.spark.mllib.util.MLUtils

object SVMWithLBFGS {

  def main(args: Array[String]) = {

    //Start the Spark context
    val conf = new SparkConf()
      .setAppName("SVM")
      .setMaster("local[*]")
    val sc = new SparkContext(conf)

    //Load examples
    val data = MLUtils.loadLibSVMFile(sc, "pubchem.svm")
    val numFeatures = data.take(1)(0).features.size //Compute number of features for LBFGS

    //Split the data
    val splits = data.randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = splits(0)
	  //adapt to LBFGS API (see Spark docs for further details)
      .map(x => (x.label, MLUtils.appendBias(x.features))) 
      .cache() 
    val test = splits(1)

    /*
     * Train the model using linear SVM. LBFGS 
     * (http://spark.apache.org/docs/latest/mllib-optimization.html#l-bfgs) is used
     * as underlying optimization algorithm here. LBFGS is a relatively new feature
     * in Spark, and it can be accessed only through low level functions.
     */
    
    //Solve the optimization problem using LBFGS. 
    //Some knowledge in Optimization is needed to actually understand this, but you can just use it as it :-)
    val (weightsWithIntercept, _) = LBFGS.runLBFGS(
      training,
      new HingeGradient(), //The Hinge objective function is what we aim to optimize in SVM
      new SimpleUpdater(), //No regularization
      //Use default paramenters for LBFGS
      numCorrections=10,
      convergenceTol=1e-4,
      maxNumIterations=100,
      regParam=0, //No regularization
      initialWeights=Vectors.dense(new Array[Double](numFeatures + 1))) //Use (0,0 ...) vector as first guess

    //Create a SVM model using the weights computed in the previous step  
    val model = new SVMModel(
      Vectors.dense(weightsWithIntercept.toArray.slice(0, weightsWithIntercept.size - 1)),
      weightsWithIntercept(weightsWithIntercept.size - 1))

    //Clear the threshold
    model.clearThreshold()

    //Compute the distance from the separating hyperplane for each of the test examples
    val scoreAndLabels = test.map { testExample =>
      val distance = model.predict(testExample.features)
      (distance, testExample.label)
    }

    //Compute the area under the ROC curve using the Spark's BinaryClassificationMetrics class
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC) //print the area under the ROC

    //Stop the Spark context
    sc.stop

  }

}
```

**Task:** try to run the previous code snippet on your machine. Do you see any improvement in the area under the ROC curve?


