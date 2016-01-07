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
