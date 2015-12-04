# Introduction to predictive modeling in Spark with applications in pharmaceutical bioinformatics
Spark is a genearal cluster computing engine for large-scale data processing. In this repository we store tutorials, and relating code, to get started with Spark predictive modeling. Even if the main focus here is pharmaceutical bioinformatics, the presented methodologies
are generally applicable, hence the following tutorials represent a good starting point for everione is interested in 
learning Spark.

## Getting started
In order to get your hands dirty, writing some code, you first need to have access to a Spark environment. Off course, having access to
cloud resources or to a Spark bare-metal installation is expensive, and time consuming in terms of set up. Hence, to get started it is
preferrable to setup a single machine environment on your computer. This will be useful even later on, when you will deploy your
Spark applications to a production environment, as you allways want to test your code locally first. Please follow this 
videotuorial in order to setup Spark on your local machine: www.youtube.com/watch?v=aB4-RD_MMf0. 

### Scala IDE
My recipe to get Spark on your local machine, is a bit unconventional if compared to other Spark's getting started tutorials. I believe that IDEs improve software developers productivity, therefore Scala IDE is the main ingredient (http://scala-ide.org/). Scala IDE, comes with an integrated Maven plugin, that can be used to pull Spark and all of its dependencies. Fianlly, using Scala IDE it is simple to
push your code to GitHub. 

### Word count
The word count problem is considered the "hello world" of big data analytics. The task performed by a word count is very simple:
given a text file, count how many times every single word occurs. 

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
      line.split(" ") //split the line in word by word. NB: we use flatMap, because we return a list
    }
    .map { word => //for each word
      (word,1) //Return a key/value tuple, with the word as key and 1 as value
    }
    .reduceByKey(_ + _) //Sum all of the value with same key
    .saveAsTextFile("food.counts.txt") //Save to a text file
    
    //Stop the Spark context
    sc.stop
    
  }

}
```

