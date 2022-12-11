import org.apache.spark.ml.regression.GeneralizedLinearRegression;
import org.apache.spark.ml.regression.GeneralizedLinearRegressionModel;
import org.apache.spark.ml.regression.GeneralizedLinearRegressionTrainingSummary;

import java.util.Arrays;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
// $example off$
public class GeneralizedLinearRegressorModel {
	public static void main(String[] args) {
		SparkSession spark = SparkSession
			      .builder()
			      .appName("JavaLogisticRegressionWithLBFGSE")
			      .master("local[*]")
			      .getOrCreate();
		
		String path = "data/Student_Marks.csv";
		Dataset<Row> dataset = spark.read().option("header", "true").csv(path);
		dataset.show(false);

		dataset = dataset.select(
				dataset.col("number_courses").cast(DataTypes.IntegerType),
				dataset.col("time_study").cast(DataTypes.DoubleType),
				dataset.col("Marks").cast(DataTypes.DoubleType)
				);

		Dataset<Row>describe = dataset.describe();
		describe.show();
		//---------------------------Splitting into train and test set---------------------//
				Dataset<Row>[] BothTrainTest = dataset.randomSplit(new double[] {0.8d,0.2d},42);
				Dataset<Row> TrainDf = BothTrainTest[0];
				Dataset<Row> TestDf = BothTrainTest[1];		

				
				
				//---------------------------Assembling Features---------------------//
				VectorAssembler assembler = new VectorAssembler()
						.setInputCols(new String[]{"number_courses", "time_study"})
						.setOutputCol("features");
				TrainDf = assembler.transform(TrainDf);
				TestDf = assembler.transform(TestDf);
				
				TrainDf.show();
				
			    GeneralizedLinearRegression glr = new GeneralizedLinearRegression()
			    		.setFeaturesCol("features")
			      .setFamily("gaussian")
			      .setMaxIter(10)
			      .setRegParam(0.3)
			      .setLabelCol("Marks");
//
//			    // Fit the model
			    GeneralizedLinearRegressionModel model = glr.fit(TrainDf);
			    Dataset<Row> predictions = model.transform(TestDf);
		    	predictions.show();
		    	
			    // Print the coefficients and intercept for generalized linear regression model
			    System.out.println("Coefficients: " + model.coefficients());
			    System.out.println("Intercept: " + model.intercept());
//
//			    // Summarize the model over the training set and print out some metrics
			    GeneralizedLinearRegressionTrainingSummary summary = model.summary();
			    System.out.println("Coefficient Standard Errors: "
			      + Arrays.toString(summary.coefficientStandardErrors()));
			    System.out.println("T Values: " + Arrays.toString(summary.tValues()));
			    System.out.println("P Values: " + Arrays.toString(summary.pValues()));
			    System.out.println("Dispersion: " + summary.dispersion());
			    System.out.println("Null Deviance: " + summary.nullDeviance());
			    System.out.println("Residual Degree Of Freedom Null: " + summary.residualDegreeOfFreedomNull());
			    System.out.println("Deviance: " + summary.deviance());
			    System.out.println("Residual Degree Of Freedom: " + summary.residualDegreeOfFreedom());
			    System.out.println("AIC: " + summary.aic());
			    System.out.println("Deviance Residuals: ");
			    summary.residuals().show();
			    // $example off
		spark.stop();
	}
}