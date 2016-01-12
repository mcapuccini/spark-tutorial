package se.uu.farmbio.tutorial

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions

object Consensus {

  def main(args: Array[String]) = {

    //Start the Spark context
    val conf = new SparkConf()
      .setAppName("Consensus")
      .setMaster("local")
    val sc = new SparkContext(conf)

    //Read DNA sequences 
    val dnaRDD = sc.textFile("dna.txt")

    /*
     * In order to be able to group residues by position
     * first we need an intermediate step where we map each residue in
     * each sequence to a key/value pair, where the key is the residue position
     * and the value is the residue itself
     */
    val keyValRDD = dnaRDD.flatMap { sequence => //for each sequence in the RDD
      sequence.zipWithIndex // zips each character with its index (see scala doc for further details)
        .map { case (residue, position) => //for each residue/position pair
            val key = position //set position as key
            val value = residue //set the residue as value
            (key, value) //return key/value pair
        }
      //N.B. we are returning a sequence, this is why we use flatMap instead of map.
    }

    //Group key/value pairs by key
    val groupedRDD = keyValRDD.groupByKey

    /*
     * Now key/value pairs (where key is a residue position, and value the residue itself) 
     * are grouped by key. Therefore, we can compute the consensus finding in the most 
     * frequent value in each of this groups. 
     */
    val consensusRDD = groupedRDD.map { 
      case (key, valuesGroup) => //for each value valueGroup (with a certain key)
        //Compute the most frequent value in the group
        val residueCounts = valuesGroup 
          .groupBy(value => value) //group equal values together 
          .map { case (value, valuesGroup) => //for each group of equal values
            (value,valuesGroup.size) // map a value/group-size pair
          }
        /*
         * At this point residueCounts contains a pair for each residue that 
         * stores the residue itself and how many times it appears at the 
         * current position (which is stored in key)
         */
        //Find the residue/count pair with max count
        val maxCountPair = residueCounts.maxBy {
          case(residue,count) => count 
        }
        /*
         * Return a pair that stores the position with its most frequent residue,
         * which is in position 1 of maxCountPair
         */
        (key, maxCountPair._1)
    }

    //Now we can format and save the consensus
    consensusRDD.sortBy { //first we sort by position (key)
      case (key, mostFrequent) =>
        key
    }.map { //now we can get rid of the position, which is no longer needed
      case (position, mostFrequent) =>
        mostFrequent
    }.saveAsTextFile("dna.consensus.txt") //finally we save to a text file

    //Stop the Spark context
    sc.stop

  }

}