# Introduction to predictive modeling in Spark with applications in pharmaceutical bioinformatics
[Spark](http://spark.apache.org/) is a genearal cluster computing engine for large-scale data processing. In this repository we store tutorials, exercises, and code snippets, to get started with Spark for predictive modeling. Even if the main focus here is pharmaceutical bioinformatics, the presented methodologies are generally applicable, hence the following tutorials represent a good starting point for everyone is interested in learning Spark. 

In this repository we code Spark programs in Scala, if you are not familiar with it please first give it a shot [here](http://www.scala-tour.com/). 

## Getting started
In order to get your hands dirty, writing and testing some Spark-based code, you will first need to access a Spark installation. Off course, cloud resources and Spark bare-metal installations are expensive options, both in terms of money and set up time. Therefore, to get started it is preferable to setup a single machine environment on your computer. This will be useful even later on, when you will deploy your Spark applications to a production environment, as you always want to test your code locally first. 

Please follow this video tutorial to setup Spark on your local machine: https://www.youtube.com/watch?v=aB4-RD_MMf0. In the video, I refer to this [pom.xml](https://github.com/mcapuccini/spark-tutorial/blob/master/spark-tutorial/pom.xml) file, you will need to open it and copy/paste some parts of it in your own configuration. 

### Scala IDE
If you followed the video tutorial you probably noticed that my recipe to get Spark on your local machine is a bit unconventional. I believe that IDEs improve software developers productivity, therefore **Scala IDE** is the main ingredient (http://scala-ide.org/). 

Scala IDE comes with an integrated **Maven** plugin, that can be used to pull Spark and all of its dependencies. Furthermore, Maven can automatically build your Scala code into production-ready jar packages. All you need to do is to configure your *pom.xml* file properly.

Another important remark is that, using Scala IDE, it is simple to sync your code to **GitHub**. 

### Word count
The **word count** problem is considered to be the *"hello world"* of big data analytics. The task performed by a word count program is very simple: *given a text file, count how many times every word occurs*. 

The main data abstraction in Spark programs is the [Resilient Distributed Dataset](http://spark.apache.org/docs/latest/programming-guide.html#resilient-distributed-datasets-rdds) (RDD), which is a distributed collection of object that can be processed in parallel, via built-in functions. The main properties of RDDs are: scalability, fault-tolerance and cacheability. The last property is of particular interest for iterative tasks such as machine learning algorithms. In the following code snippet, we use an RDD, along with its built-in functions, to implement a parallel *word count* program. Since we are using RDDs, such implementation will be out-of-the-box scalable and fault tolerant.  

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
 3. Remember that we need the consensus to be sorted position-wise. Therefore, don't forget to keep track of the position of each most frequent residue, hence you will be able to use [sortByKey](http://spark.apache.org/docs/latest/programming-guide.html#working-with-key-value-pairs) in order to sort your result. 

**Solution:** you can give a look to the solution [here](https://github.com/mcapuccini/spark-tutorial/blob/master/spark-tutorial/src/main/scala/se/uu/farmbio/tutorial/Consensus.scala), but first try it yourself :smirk:

##Predictive modelling in Spark
In **predictive modelling**, basing on previous observations, we aim to build a statistical model to predict the future behaviour of a certain system. A *predictive model* is a function to predict future outcomes, basing on a number of *features*, which provide the means to describe an event in the system that we aim to model. In *pharmaceutical bioinformatics*,  predictive modelling is used in order to predict molecular behaviours, such as *binding affinity* or *toxicology*. [Molecular descriptors](https://en.wikipedia.org/wiki/Molecular_descriptor) such as [log P](https://en.wikipedia.org/wiki/Log_P), [molar refractivity](https://en.wikipedia.org/wiki/Molar_refractivity), [dipole moment](https://en.wikipedia.org/wiki/Molecular_dipole_moment), [polarizability](https://en.wikipedia.org/wiki/Polarizability), and [molecular signatures](http://www.ncbi.nlm.nih.gov/pubmed/15032522),  are mostly used as features to make up such predictive models. 

In predictive modelling, we call **training examples** the previous observations that we use in order to *train* a predictive model. Each of the training examples stores a system outcome, that we call *label*, and a features vector that describes that outcome. Of course, choosing the right set of features to describe the system that we are trying to model is crucial. An example follows. 

**Training examples**  

| Toxicity (label)  | Log P (feature 1) | Polarizability (feature 2) | ... |  Dipole moment (fearure n) |
| -------------  | ----------------- | -------------------------- | ----| ----  |
| 1.0 (yes)      | 0.450...          | 1.309...                   | ... | 1.297 |
| 0.0 (no)       | 0.346...          | 3.401...                   | ... | 0.281 |
| 1.0 (yes)      | 4.446...          | 2.431...                   | ... | 6.741 |
| 0.0 (no)       | 3.306...          | 0.473...                   | ... | 1.365 |
| ... | ... | ... | ... | ... |


Basing on the training examples, a variety of **machine leaning** algorithms can be used in order to train a model. [Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine) (SVM) have been successfully used in order to build predictive models in pharmaceutical bioinformatics.  A [linear SVM implementation](http://spark.apache.org/docs/latest/mllib-linear-methods.html#linear-support-vector-machines-svms) is provided by the [Spark MLlib](http://spark.apache.org/mllib/) package. This enables predictive modelling for pharmaceutical bioinformatics over big training datasets. The following code snippet shows how to train a predictive model using the Spark SVM implementation. 
 
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
    val distAndLabels = test.map { testExample =>
      val distance = model.predict(testExample.features) 
      (distance, testExample.label) 
      /*
       * N.B. we keep track of the label, since we want to compute sensitivity and
       * specificity (in order to plot the ROC curve)
       */
    }

    //Compute the area under the ROC curve using the Spark's BinaryClassificationMetrics class
    val metrics = new BinaryClassificationMetrics(distAndLabels)
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC) //print the area under the ROC

    //Stop the Spark context
    sc.stop

  }

}
```

**Maven Spark MLlib dependency:** in order to run the previous code you need to add the Spark MLlib dependency to your *pom.xml* file.

```xml
<dependencies>
	...
	<dependency>
		<groupId>org.apache.spark</groupId>
		<artifactId>spark-mllib_2.10</artifactId>
		<version>1.6.0</version>
	</dependency>
	...
<dependencies>
```

In the previous program, the [pubchem.svm](https://raw.githubusercontent.com/mcapuccini/spark-tutorial/master/spark-tutorial/pubchem.svm) file is used as input. This file contains examples where the labels represent toxicology, and the features vector a molecular signature. Let's give a look to a bunch of examples in *pubchem.svm*: 

```no-highlight
0.0 11234:2.0 11482:1.0 12980:1.0 13434:1.0 13858:1.0 23167:1.0 26439:2.0 30078:1.0 30377:1.0 38851:1.0 39621:1.0 41080:2.0 48528:1.0 54325:1.0 54447:2.0 65490:1.0 65991:1.0 71163:1.0 74579:3.0 81127:2.0 86247:1.0 92687:1.0 103188:1.0 103437:2.0 106964:1.0 114196:2.0 121864:1.0 125845:1.0 126651:1.0 132509:1.0 138301:1.0 143915:1.0 145561:1.0 146537:1.0 151499:2.0 152885:1.0 156424:1.0 160914:1.0 163411:2.0 167790:2.0 176961:7.0 178108:2.0 181972:1.0 182021:1.0
1.0 3639:2.0 4450:1.0 5494:2.0 9998:1.0 13951:1.0 18213:1.0 18323:1.0 18797:1.0 22797:1.0 23347:1.0 26441:1.0 26526:2.0 30605:2.0 44244:1.0 54325:2.0 56124:2.0 62618:2.0 67306:1.0 67926:1.0 68056:1.0 68646:1.0 73422:2.0 74579:6.0 76833:1.0 81127:2.0 85885:2.0 92647:1.0 93882:1.0 94432:1.0 96374:2.0 97697:2.0 105394:1.0 106301:1.0 106411:1.0 107633:1.0 111281:1.0 111394:1.0 113160:1.0 118188:1.0 119006:1.0 122468:2.0 136300:1.0 136849:1.0 144309:2.0 149235:1.0 149439:1.0 149956:2.0 158381:2.0 163411:1.0 165703:1.0 175370:2.0 176961:13.0 181732:1.0
0.0 20307:1.0 23415:1.0 24337:1.0 36469:1.0 37715:1.0 41512:1.0 45035:1.0 48936:1.0 53031:1.0 54447:2.0 58285:1.0 66077:1.0 69559:1.0 70494:1.0 79582:1.0 90338:1.0 91787:1.0 97697:1.0 101222:1.0 102151:1.0 102692:1.0 113325:1.0 114349:1.0 116804:1.0 122408:1.0 122549:1.0 126712:1.0 126904:1.0 137469:1.0 138146:1.0 143846:1.0 144149:1.0 145926:1.0 157873:1.0 163411:1.0 175272:1.0 176882:1.0 176942:1.0
1.0 2916:1.0 3639:1.0 6834:1.0 9861:1.0 10642:1.0 12333:1.0 19635:1.0 24916:2.0 27547:1.0 28559:1.0 31499:1.0 35183:1.0 41127:1.0 41916:1.0 46296:1.0 48528:1.0 54447:1.0 54860:1.0 56319:1.0 64081:1.0 65740:1.0 68516:1.0 74579:1.0 77274:1.0 79639:1.0 83151:2.0 97238:1.0 104627:2.0 106964:1.0 110246:1.0 117150:1.0 122408:1.0 124873:1.0 131661:1.0 132518:1.0 135412:1.0 135571:1.0 153997:1.0 163026:1.0 163411:2.0 164716:1.0 170384:1.0 172966:1.0 176041:1.0 176961:5.0 177733:1.0
```

The *pubchem.svm* file encodes the examples in the [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) format. In such format each example is stored in a line, where the first number represents the label and the remaining string the features vector in a sparse representation. *LIBSVM format* encodes the feature vector with a series of *X:Y* entries, where *X* represents a position in the vector and *Y* the value at that position. Notice how molecular signatures in *pubchem.svm* result to highly sparse vectors, where over thousands of positions only few actually store a value.

Using molecular signatures, since the data is very sparse, it is important to load the examples in your program using a sparse vector representation. Fortunately, *Spark MLlib* loads *LIBSVM* files using the [LabelledPoint](http://spark.apache.org/docs/latest/mllib-data-types.html#labeled-point) data type, which can store a feature vector in sparse representation. 

**Task:** try to run the previous code snippet on your machine, how good is the area under the ROC curve?

###SVM with LBFGS optimization
In the previous code snippet we trained the model using SVM, with the default [Stocastic Gradient Descent](http://spark.apache.org/docs/latest/mllib-optimization.html#stochastic-gradient-descent-sgd) (SGD) optimization algorithm. This happens to work poorly with molecular datasets, because SGD is designed to deal with really huge data (e.g. streams of tweets). However, Spark provides [LBFGS](http://spark.apache.org/docs/latest/mllib-optimization.html#l-bfgs) as an alternative to SGD.  Hence, the previous code can be adapted to use LBFGS in order to improve the model performance. 

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
    val distAndLabels = test.map { testExample =>
      val distance = model.predict(testExample.features)
      (distance, testExample.label)
    }

    //Compute the area under the ROC curve using the Spark's BinaryClassificationMetrics class
    val metrics = new BinaryClassificationMetrics(distAndLabels)
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC) //print the area under the ROC

    //Stop the Spark context
    sc.stop

  }

}
```

**Task:** try to run the previous code snippet on your machine. Do you see any improvement in the area under the ROC curve?

##Exercise 2: build a toxicology prediction model using Logistic Regression

Spark offers some alternatives to SVM. One of these is [Logistic Regression](http://spark.apache.org/docs/latest/mllib-linear-methods.html#logistic-regression).

**Task:** starting from the previous code snippets, write a Spark program to train a toxicology prediction model using logistic regression instead of SMV. Which does perform best?

**Hint:** Remember that SGD doesn't perform good with molecular datasets, therefore you need to use LBFGS instead. Fortunately the community implemented [LogisticRegressionWithLBFGS](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS), hence you don't have to set up the optimization problem manually (like we did in the SVM example).

**Solution:** you can give a look to the solution [here](https://github.com/mcapuccini/spark-tutorial/blob/master/spark-tutorial/src/main/scala/se/uu/farmbio/tutorial/LogisticRegressionExample.scala). As before, try it yourself first!

##Conformal prediction
In pharmaceutical bioinformatics, assigning a *confidence level* to predictions plays a major role. In fact, if you think to the toxicology models that we built in the previous examples, due to security reasons, the predictions will be useful in practice only if we can assign to them a valid likelihood of correctness. 

[Conformal prediction](http://www.alrw.net/articles/03.pdf) is a mathematical framework that allows to assign a confidence level to each prediction, basing on a solid background theory. This contrasts to current best-practices (e.g. [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_%28statistics%29)) where an overall confidence level for predictions on new examples is hedged basing on previous performance. Given a user-specified significance level 𝜺, instead of producing a single prediction *l*, a **conformal predictor** outputs a **prediction set** *{l<sub>1</sub>, l<sub>2</sub> ... l<sub>n</sub>}*. The major advantage of this approach is that, within the mathematical framework, there is proof that *the true label for a new example will be in the prediction set with probability at least 1 - 𝜺*. This enables to assign confidence *1 - 𝜺* to the produced prediction sets.

**N.B.** Conformal prediction doesn't substitute any of the training algorithms that we explained previously, but rather represent a methodology to apply on top of any machine learning algorithm, to assign confidence levels to predictions. 

###Validity and Efficiency
We say that a conformal predictor is **valid**, for a certain significance level  𝜺, when the observed error rate is at most 𝜺. This is always true **on average**, by construction. However, a conformal predictor that outputs too big prediction sets is unuseful. For instance, in toxicology modelling we are just interested in **singleton prediction sets**; empty *{}* or both classes *{toxic, non-toxic}* prediction sets means that, at the chosen significance level, the prediction is *rejected*. 

In [binary classification](https://en.wikipedia.org/wiki/Binary_classification) (e.g. toxicology predictive modelling), we define the **efficiency** of a conformal predictor, with respect to a certain significance level, as the observed *singleton prediction set rate*. This measure tells us how useful a conformal predictior is for certain significance level. 

###Conformal prediction in Spark
In the [Pharmaceutical Bioscience Department](http://farmbio.uu.se/) at Uppsala University (Sweden), we implemented a Spark-based Conformal Prediction package, to enable predictive modelling with confidence over big datasets. If you aim to use that in one of your Spark projects, you need to add the following repository and dependency to the *pom.xml* file.

```xml 
<repositories>
    ...
    <repository>
        <id>pele.farmbio.uu.se</id>
        <url>http://pele.farmbio.uu.se/artifactory/libs-snapshot</url>
    </repository>
    ...
</repositories>

<dependencies>
...
    <groupId>se.uu.farmbio</groupId>
        <artifactId>cp</artifactId>
        <version>0.0.1-SNAPSHOT</version>
    </dependency>
...
</dependencies>
```

In the following code snippet we use our package to train a conformal predictor, using SVM as machine leaning algorithm.

```scala 
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
```
**Task:** try to run the previous code snippet on your machine. How does the conformal predictor perform on the training examples?


##Exercise 3: build a toxicology prediction model with Logistic Regression based Conformal Prediction
Logistic regression is another machine learning algorithm built-in Spark. Since conformal prediction applies to any machine learning algorithm, it is interesting to investigate on how the efficiency changes using logistic regression instead of SVM.

**Task:** starting from the previous code snippet, write a Spark program to train a toxicology prediction model using a logistic regression based conformal prediction. How does the performance compare with the SVM based implementation?

**Hint:** to get the job done quickly, you just need to switch the [LogisticRegression](https://github.com/mcapuccini/spark-cp/blob/master/cp/src/main/scala/se/uu/farmbio/cp/alg/LogisticRegression.scala) class to the SVM class, when you train the inductive conformal predictor. 

**Solution:** you can give a look to the solution [here](https://github.com/mcapuccini/spark-tutorial/blob/master/spark-tutorial/src/main/scala/se/uu/farmbio/tutorial/CPWithLR.scala). As usual, it's good to try it yourself first. 