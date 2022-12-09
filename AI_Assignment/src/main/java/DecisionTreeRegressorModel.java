
import org.apache.spark.ml.regression.DecisionTreeRegressor;
import org.apache.spark.ml.regression.DecisionTreeRegressionModel;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
// $example off$
public class DecisionTreeRegressorModel {
	public static void main(String[] args) {
		SparkSession spark = SparkSession
			      .builder()
			      .appName("JavaLogisticRegressionWithLBFGSE")
			      .master("local[*]")
			      .getOrCreate();
		
		
		
		//---------------------------Loading Dataset---------------------//
		String path = "data/Student_Marks.csv";
		Dataset<Row> df = spark.read().option("header", "true").csv(path);
		df.show(false);

		df = df.select(
				df.col("number_courses").cast(DataTypes.IntegerType),
				df.col("time_study").cast(DataTypes.DoubleType),
				df.col("Marks").cast(DataTypes.DoubleType)
				);

		Dataset<Row>describe = df.describe();
		describe.show();

		

		//---------------------------Splitting into train and test set---------------------//
		Dataset<Row>[] BothTrainTest = df.randomSplit(new double[] {0.8d,0.2d},42);
		Dataset<Row> TrainDf = BothTrainTest[0];
		Dataset<Row> TestDf = BothTrainTest[1];		

		
		
		//---------------------------Assembling Features---------------------//
		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(new String[]{"number_courses", "time_study"})
				.setOutputCol("features");
		TrainDf = assembler.transform(TrainDf);
		TestDf = assembler.transform(TestDf);
		
		TrainDf.show();

		
		
		//---------------------------Model Training---------------------//
		DecisionTreeRegressor dt = new DecisionTreeRegressor()
	    		.setFeaturesCol("features")
	    		.setMaxDepth(15)
	    		.setMinInstancesPerNode(2)
	    		.setLabelCol("Marks");
	
		DecisionTreeRegressionModel model = dt.fit(TrainDf);
	    	Dataset<Row> predictions = model.transform(TestDf);
	    	predictions.show();
	    
	    
	  	//---------------------------Evaluating Predictions---------------------//
	    
	  	//---------------------------RMSE---------------------//
	    	RegressionEvaluator evaluator_rmse = new RegressionEvaluator()
	    		.setLabelCol("Marks")
	    		.setPredictionCol("prediction")
	    		.setMetricName("rmse");
	    	Double rmse = evaluator_rmse.evaluate(predictions);
	    
	    
	  	//---------------------------MSE---------------------//
	    	RegressionEvaluator evaluator_mse = new RegressionEvaluator()
	    		.setLabelCol("Marks")
	    		.setPredictionCol("prediction")
	    		.setMetricName("mse");
	    	Double mse = evaluator_mse.evaluate(predictions);
	    
	    
	  	//---------------------------R2---------------------//
	    	RegressionEvaluator evaluator_r2 = new RegressionEvaluator()
	    		.setLabelCol("Marks")
	    		.setPredictionCol("prediction")
	    		.setMetricName("r2");
	    	Double r2 = evaluator_r2.evaluate(predictions);
	    
	    
	  	//---------------------------MAE---------------------//
	    	RegressionEvaluator evaluator_mae = new RegressionEvaluator()
	    		.setLabelCol("Marks")
	    		.setPredictionCol("prediction")
	    		.setMetricName("mae");
	    	Double mae = evaluator_mae.evaluate(predictions);
	    
	    
	    	System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);
	    	System.out.println("Mean Squared Error (MSE) on test data = " + mse);
	    	System.out.println("Root Squared (R2) on test data = " + r2);
	    	System.out.println("Mean Absolute Error (MAE) on test data = " + mae);


		spark.stop();
	}
}
