package JavaExample;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;

// $example on$
import scala.Tuple2;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
// $example off$

public class JavaLogisticRegressionWithLBFGSE {
  public static void main(String[] args) {
    SparkConf conf = new SparkConf().setAppName("JavaLogisticRegressionWithLBFGSExample");
    SparkContext sc = new SparkContext(conf);
    // $example on$
    String path = "/SimpleProject123/resources/Student_Marks.csv";
    JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc, path).toJavaRDD();

    // Split initial RDD into two... [70% training data, 30% testing data].
    JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[] {0.7, 0.3}, 11L);
    JavaRDD<LabeledPoint> training = splits[0].cache();
    JavaRDD<LabeledPoint> test = splits[1];

    // Run training algorithm to build the model.
    LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
      .setNumClasses(10)
      .run(training.rdd());

    // Compute raw scores on the test set.
    JavaPairRDD<Double, Double> predictionAndLabels = test.mapToPair(p ->new Tuple2<>(model.predict(p.features()), p.label()));

    // Get evaluation metrics.
    MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
    double accuracy = metrics.accuracy();
    System.out.println("Accuracy = " + accuracy);

    // Save and load model
    model.save(sc, "target/tmp/javaLogisticRegressionWithLBFGSModel");
    LogisticRegressionModel sameModel = LogisticRegressionModel.load(sc,
      "target/tmp/javaLogisticRegressionWithLBFGSModel");
    // $example off$

    sc.stop();
  }
}
 