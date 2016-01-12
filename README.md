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

In predictive modelling, we call **training examples** the previous observations that we use in order to *train* the predictive model. Each of the training examples stores a system outcome, that we call *label*, and a features vector that describes that outcome. Of course, choosing the right set of features to describe the behaviour we are trying to model is crucial. An example follows. 

**Training examples**  

| Toxicity (label)  | Log P (feature 1) | Polarizability (feature 2) | ... |  Dipole moment (fearure n) |
| -------------  | ----------------- | -------------------------- | ----| ----  |
| 1.0 (yes)      | 0.450...          | 1.309...                   | ... | 1.297 |
| 0.0 (no)       | 0.346...          | 3.401...                   | ... | 0.281 |
| 1.0 (yes)      | 4.446...          | 2.431...                   | ... | 6.741 |
| 0.0 (no)       | 3.306...          | 0.473...                   | ... | 1.365 |
| ... | ... | ... | ... | ... |


Basing on the training examples, a variety of **machine leaning** algorithms can be used in order to train a model. [Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine) (SVM) have been successfully used in order to build predictive models in Pharmaceutical Bioinformatics.  A linear SVM implementation is provided by the [Spark MLlib](http://spark.apache.org/mllib/) package. This enables  





