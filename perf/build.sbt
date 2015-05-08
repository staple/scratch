name := "tfocs-perf-test"

version in ThisBuild := "1.0-SNAPSHOT"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.3.1" % "provided",
  "org.apache.spark" %% "spark-mllib" % "1.3.1" % "provided"
)

scalariformSettings
