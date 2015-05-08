/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.tfocs

import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.linalg.{ Vectors, Vector }
import org.apache.spark.mllib.linalg.BLAS

import scala.concurrent._
import ExecutionContext.Implicits.global

/**
 * Trait for smooth functions.
 */
trait SmoothFunction[X] {
  /**
   * Evaluates this function at x and returns the function value and its gradient based on the mode
   * specified.
   */
  def apply(x: X, mode: Mode): Value[Double, X]

  /**
   * Evaluates this function at x.
   */
  def apply(x: X): Double = apply(x, Mode(f = true, g = false)).f.get

  def two(x: X, data: RDD[Vector], features: Int): (Double, Vector)

  def future(x: X): (Future[Double], RDD[Double])
}

class SquaredErrorRDDDouble(x0: RDD[Double]) extends SmoothFunction[RDD[Double]] {

  x0.cache()

  override def apply(x: RDD[Double], mode: Mode): Value[Double, RDD[Double]] = {
    val g = x.zip(x0).map(x => x._1 - x._2)
    if (mode.f && mode.g) g.cache()
    val f = if (mode.f) Some(g.treeAggregate(0.0)((sum, x) => sum + x * x, _ + _) / 2.0) else None
    Value(f, Some(g))
  }

  override def two(x: RDD[Double], data: RDD[Vector], features: Int): (Double, Vector) = {
    x.zip(x0).map(x => x._1 - x._2).zip(data).treeAggregate((0.0, Vectors.zeros(features)))(
      seqOp = (c, v) => {
        BLAS.axpy(v._1, v._2, c._2)
        (v._1 * v._1 / 2.0 + c._1, c._2)
      },
      combOp = (c1, c2) => {
        BLAS.axpy(1.0, c2._2, c1._2)
        (c1._1 + c2._1, c1._2)
      })
  }

  override def future(x: RDD[Double]): (Future[Double], RDD[Double]) = {
    val g = x.zip(x0).map(x => x._1 - x._2)
    g.cache()
    val f: Future[Double] = scala.concurrent.future {
      g.treeAggregate(0.0)((sum, z) => sum + z * z, _ + _) / 2.0
    }
    (f, g)
  }
}

