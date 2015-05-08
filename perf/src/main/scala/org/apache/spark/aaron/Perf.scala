package org.apache.spark.aaron

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.mllib.random._
import org.apache.spark.mllib.random.RandomDataGenerator
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.linalg.BLAS
import org.apache.spark.rdd.RDD

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._

import org.apache.spark.mllib.linalg.{DenseVector, Vectors, Vector}

import org.apache.spark.mllib.optimization._

object Perf {

  val PARTITIONS = 192

  def gc() = { sc.parallelize(1 to PARTITIONS, PARTITIONS).foreach(x => System.gc()) }

  var sc: SparkContext = _

  class L1UpdaterBLAS extends Updater {
    override def compute(
        weightsOld: Vector,
        gradient: Vector,
        stepSize: Double,
        iter: Int,
        regParam: Double): (Vector, Double) = {
      val thisIterStepSize = stepSize / math.sqrt(iter)
      // Take gradient step
      // val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
      // brzAxpy(-thisIterStepSize, gradient.toBreeze, brzWeights)
      val x = weightsOld.copy
      BLAS.axpy(-thisIterStepSize, gradient, x)

      // Apply proximal operator (soft thresholding)
      val shrinkageVal = regParam * thisIterStepSize
      var i = 0
      val g = shrinkageVal match {
        case 0.0 => x
        case _ => new DenseVector(x.toArray.map(x => x * (1.0 - math.min(shrinkageVal / math.abs(x), 1.0))))
      }

      (g, Vectors.norm(g, 1) * regParam)
    }
  }

  def makeData(rows: Int, cols: Int, tag: String) = {
    class GaussianMatrix(val numFeatures: Int) extends RandomDataGenerator[Vector] {

      private val rng = new java.util.Random()

      override def nextValue(): Vector = {
        Vectors.dense(Array.fill[Double](numFeatures)(rng.nextGaussian()))
    }

    override def setSeed(seed: Long) {
      rng.setSeed(seed)
    } 

    override def copy(): GaussianMatrix =
      new GaussianMatrix(numFeatures)

    }

    val seed = 92
    val matrix = RandomRDDs.randomRDD(sc, new GaussianMatrix(cols), rows, PARTITIONS, seed)
    matrix.persist(StorageLevel.MEMORY_AND_DISK)
    matrix.count
    println("matrix")
    println(matrix.partitions.length)

    // normalize columns
    val norms = matrix.map(x => Vectors.dense(x.toArray.map(x => x * x))).reduce( (a, b) => {
    Vectors.dense(a.toArray.zip(b.toArray).map(x => x._1 + x._2))
    }).toArray.map(x => math.sqrt(x))

    val normed = matrix.map(x => Vectors.dense(x.toArray.zip(norms).map(x => x._1 / x._2)))
    println("normed")
    println(normed.partitions.length)
    normed.persist(StorageLevel.MEMORY_AND_DISK)
    normed.count

    matrix.unpersist(true)

    val rng = new java.util.Random(seed)

    val nonzeroWeights = Array.fill(cols / 10){ rng.nextGaussian() }
    val orderedWeights = nonzeroWeights ++ Array.fill(cols - cols / 10)(0.0)
    val weights = Vectors.dense(scala.util.Random.shuffle(orderedWeights.toList).toArray)

    val b_original = normed.map(x => BLAS.dot(x, weights))
    println("b_original")
    println(b_original.partitions.length)
    val snr = 30

    val bo = b_original.collect()
    val bonorm = Vectors.norm(Vectors.dense(bo), 2)

    val sigma = math.pow(10, (10 * math.log10( math.pow( bonorm, 2) / cols) - snr) / 20 )

    val term = math.pow(10, (10 * math.log10( bo.map(x => math.abs(x) * math.abs(x)).reduce(_ + _) / bo.length) - snr) / 20)

    val b = b_original.map(x => x + rng.nextGaussian() * term)
    println("b")
    println(b.partitions.length)

    val data = b.zip(normed)

    data.saveAsObjectFile(s"hdfs://ec2-52-24-61-10.us-west-2.compute.amazonaws.com:9000/aaron-perf-data-$tag")

    val lambda = 2 * sigma * math.sqrt(2 * math.log(cols))

    sc.parallelize(List(lambda)).saveAsObjectFile(s"hdfs://ec2-52-24-61-10.us-west-2.compute.amazonaws.com:9000/aaron-perf-lambda-$tag")

    normed.unpersist()
  }

  def main(args: Array[String]) {

      val sizes = List(12500, 25000, 50000, 100000)

      //for (rows <- sizes; cols <- sizes if !(rows == 100000 && cols == 100000)) {

      val rows = 25000
      val cols = 12500
  val tag = s"$rows-$cols"

    val gd = new ArrayBuffer[Double](4)
    val agd = new ArrayBuffer[Double](4)
    val agdo = new ArrayBuffer[Double](4)
    val tfocs = new ArrayBuffer[Double](4)
    val tfocso = new ArrayBuffer[Double](4)
    val tfocsf = new ArrayBuffer[Double](4)


    val times = new ArrayBuffer[Double](4)

    val mode: Integer = args(1).toInt
    //for (mode <- 0 to 5) {

    {

    val sparkConf = new SparkConf()
    sparkConf.setAppName("Perf2")
    sc = new SparkContext(sparkConf)


      {


    val data = try {
      val d = sc.objectFile[(Double, Vector)](s"hdfs://ec2-52-24-61-10.us-west-2.compute.amazonaws.com:9000/aaron-perf-data-$tag")
      println(d.first._1)
      d
    } catch {
      case e: org.apache.hadoop.mapred.InvalidInputException =>
      makeData(rows, cols, tag)
        val d = sc.objectFile[(Double,Vector)](s"hdfs://ec2-52-24-61-10.us-west-2.compute.amazonaws.com:9000/aaron-perf-data-$tag")
              println(d.first._1)
        d
    }
    println(data.partitions.length)

    val lambda = sc.objectFile[Double](s"hdfs://ec2-52-24-61-10.us-west-2.compute.amazonaws.com:9000/aaron-perf-lambda-$tag").first
    println(lambda)

    // println(bonorm)
    // println(sigma)
    // println(term)
    // println(lambda)

    val b = data.map(_._1)
    val A = data.map(_._2)

    val scaledData = b.map(_ * math.sqrt(rows)).zip(A.map(x => Vectors.dense(x.toArray.map(_ * math.sqrt(rows)))))

  if (mode <= 2) {
    println(scaledData.cache.count)
    
  }
  else {
    println(A.cache.count)
    println(b.cache.count)
  }

    // Use an over precise convergence tol to force numIterations to be the
    // same in all test cases.
    val numIterations = 30
    val convergenceTol = 1e-20

    val gradient = new LeastSquaresGradient()
    val updater = new L1Updater()
    val stepSize = 1.0 / 10.0
    val x0 = Vectors.dense(Array.fill(cols)(0.0))

    import org.apache.spark.mllib.tfocs._
    import org.apache.spark.mllib.tfocs.VectorSpace._

      for (i <- 1 to 4) {

	if (mode == 0){
    gc()
      val t0 = System.nanoTime()
      val res = GradientDescent.runMiniBatchSGD(scaledData,
                                                gradient,
                                                updater,
                                                stepSize,
                                                numIterations,
                                                lambda,
                                                1.0,
                                                x0)
      val t1 = System.nanoTime()
      gd.append(t1 - t0)
      times.append(t1 - t0)
      // scala.tools.nsc.io.File("gd_w.csv").writeAll(res._1.toArray.mkString("\n"))
      // scala.tools.nsc.io.File("gd.csv").writeAll(res._2.toArray.mkString("\n"))
      // scala.tools.nsc.io.File("gd_time.csv").writeAll((t1 - t0).toString)
    }

	if (mode == 1){
    gc()
      val t0 = System.nanoTime()
      val res = AcceleratedGradientDescent.run(scaledData,
                                               gradient,
                                               updater,
                                               convergenceTol,
                                               numIterations,
                                               lambda,
                                               x0,
                                               10.0,
                                               Double.PositiveInfinity,
                                               0.5,
                                               0.9,
                                               true)
      val t1 = System.nanoTime()
    	agd.append(t1 - t0)
      times.append(t1 - t0)
      // scala.tools.nsc.io.File("agd_w.csv").writeAll(res._1.toArray.mkString("\n"))
      // scala.tools.nsc.io.File("agd.csv").writeAll(res._2.toArray.mkString("\n"))
      // scala.tools.nsc.io.File("agd_time.csv").writeAll((t1 - t0).toString)
    }

	if (mode == 2){
    gc()
      val t0 = System.nanoTime()
      val res = AcceleratedGradientDescentOptimized.run(scaledData,
                                                        gradient,
                                                        new L1UpdaterBLAS(),
                                                        convergenceTol,
                                                        numIterations,
                                                        lambda,
                                                        x0,
                                                        10.0,
                                                        Double.PositiveInfinity,
                                                        0.5,
                                                        0.9,
                                                        true)
      val t1 = System.nanoTime()
    	agdo.append(t1 - t0)
      times.append(t1 - t0)
      // scala.tools.nsc.io.File("agdo_w.csv").writeAll(res._1.toArray.mkString("\n"))
      // scala.tools.nsc.io.File("agdo.csv").writeAll(res._2.toArray.mkString("\n"))
      // scala.tools.nsc.io.File("agdo_time.csv").writeAll((t1 - t0).toString)
    }

	if (mode == 3){
    gc()
      val t0 = System.nanoTime()
      val res = TFOCS.minimize(new SquaredErrorRDDDouble(b),
                               new ProductRDDVector(A),
                               new L1ProxVector(lambda),
                               x0,
                               numIterations,
                               convergenceTol)

      val t1 = System.nanoTime()
    	tfocs.append(t1 - t0)
      times.append(t1 - t0)
      // scala.tools.nsc.io.File("tfocs_w.csv").writeAll(res._1.toArray.mkString("\n"))
      // scala.tools.nsc.io.File("tfocs.csv").writeAll(res._2.toArray.mkString("\n"))
      // scala.tools.nsc.io.File("tfocs_time.csv").writeAll((t1 - t0).toString)
    }

	if (mode==4){
    gc()
      val t0 = System.nanoTime()
      val res = TFOCS_optimized.minimize(new SquaredErrorRDDDouble(b),
                                         new ProductRDDVector(A),
                                         new L1ProxVector(lambda),
                                         A,
                                         cols,
                                         x0,
                                         numIterations,
                                         convergenceTol)

      val t1 = System.nanoTime()
    	tfocso.append(t1 - t0)
      times.append(t1 - t0)
      // scala.tools.nsc.io.File("tfocso_w.csv").writeAll(res._1.toArray.mkString("\n"))
      // scala.tools.nsc.io.File("tfocso.csv").writeAll(res._2.toArray.mkString("\n"))
      // scala.tools.nsc.io.File("tfocso_time.csv").writeAll((t1 - t0).toString)
    }

	if (mode==5){
    gc()
      val t0 = System.nanoTime()
      val res = TFOCS_future.minimize(new SquaredErrorRDDDouble(b),
                                      new ProductRDDVector(A),
                                      new L1ProxVector(lambda),
                                      x0,
                                      numIterations,
                                      convergenceTol)

      val t1 = System.nanoTime()
    	tfocsf.append(t1 - t0)
      times.append(t1 - t0)
      // scala.tools.nsc.io.File("tfocsf_w.csv").writeAll(res._1.toArray.mkString("\n"))
      // scala.tools.nsc.io.File("tfocsf.csv").writeAll(res._2.toArray.mkString("\n"))
      // scala.tools.nsc.io.File("tfocsf_time.csv").writeAll((t1 - t0).toString)
    }

      }


      }
    sc.stop()
      }

  // val times: List[List[Any]] = List(List("gd", "agd", "agdo", "tfocs", "tfocso", "tfocsf")) ++ List(gd, agd, agdo, tfocs, tfocso, tfocsf).transpose
  //     scala.tools.nsc.io.File(s"new-30-times-$tag.csv").writeAll(times.map(x => x.toArray.mkString(",")).mkString("\n"))

      scala.tools.nsc.io.File(s"new-30-times-$mode-$tag.csv").writeAll(times.mkString("\n"))


  }
}
