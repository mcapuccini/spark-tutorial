# Introduction to predictive modeling in Spark with applications in pharmaceutical bioinformatics
**Spark** is a genearal cluster computing engine for large-scale data processing. In this repository we store tutorials, and relating code examples, to get started with Spark for predictive modeling. Even if the main focus here is pharmaceutical bioinformatics, the presented methodologies are generally applicable, hence the following tutorials represent a good starting point for everyone is interested in learning Spark.

## Getting started
In order to get your hands dirty, writing and testing some Spark-based code, you will first need to access a Spark installation. Off course, cloud resources and Spark bare-metal installations are expensive options, both in terms of money and set up time. Therefore, to get started it is preferable to setup a single machine environment on your computer. This will be useful even later on, when you will deploy your Spark applications to a production environment, as you always want to test your code locally first. 

Please follow this video tutorial to setup Spark on your local machine: https://www.youtube.com/watch?v=aB4-RD_MMf0. 

### Scala IDE
If you followed the video tutorial you probably noticed that my recipe to get Spark on your local machine, is a bit unconventional. I believe that IDEs improve software developers productivity, therefore **Scala IDE** is the main ingredient (http://scala-ide.org/). 

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

##Consensus sequence in Spark
Computing the **consensus** for a sequence alignment is an interesting problem in bioinfarmatics (https://en.wikipedia.org/wiki/Consensus_sequence). Given an alignment of several sequences, we want to find the consensus sequence, that is *the sequence that has the most frequent residue in each position of the alignment*.

>**Example**

>*Sequence 1*: A C C T G

>*Sequence 2*: G G G T C

>*Sequence 3*: A C A T C

>*Sequence 4*: C C G G C

>*Consensus* : **A C G T C**




