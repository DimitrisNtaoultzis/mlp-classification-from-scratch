import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.io.PrintWriter;
import java.io.IOException;

public class DatasetT{
	
	static int classify(double x1, double x2){
		
		double d1 = (x1 - 0.5)*(x1 - 0.5) + (x2 - 0.5)*(x2 - 0.5);
		double d2 = (x1 - 1.5)*(x1 - 1.5) + (x2 - 0.5)*(x2 - 0.5);
		double d3 = (x1 - 0.5)*(x1 - 0.5) + (x2 - 1.5)*(x2 - 1.5);
		double d4 = (x1 - 1.5)*(x1 - 1.5) + (x2 - 1.5)*(x2 - 1.5);
		double d5 = (x1 - 1.0)*(x2 - 1.0);
		
		if(d1 < 0.2 && x1 > 0.5 && x2 > 0.5){
			return 1;
		}//1
		
		if(d1 < 0.2 && x1 < 0.5 && x2 > 0.5){
			return 2;
		}//2
			
		if(d1 < 0.2 && x1 > 0.5 && x2 < 0.5){
			return 2;
		}//3

		if(d1 < 0.2 && x1 < 0.5 && x2 < 0.5){
			return 1;
		}//4
		
		if(d2 < 0.2 && x1 > 1.5 && x2 > 0.5){
			return 1;
		}//5
		
		if(d2 < 0.2 && x1 < 1.5 && x2 > 0.5){
			return 2;
		}//6
		
		if(d2 < 0.2 && x1 > 1.5 && x2 < 0.5){
			return 2;
		}//7
		
		if(d2 < 0.2 && x1 < 1.5 && x2 < 0.5){
			return 1;
		}//8
		
		if(d3 < 0.2 && x1 > 0.5 && x2 > 1.5){
			return 1;
		}//9
		
		if(d3 < 0.2 && x1 < 0.5 && x2 > 1.5){
			return 2;
		}//10
		
		if(d3 < 0.2 && x1 > 0.5 && x2 < 1.5){
			return 2;
		}//11
		
		if(d3 < 0.2 && x1 < 0.5 && x2 < 1.5){
			return 1;
		}//12
		
		if(d4 < 0.2 && x1 > 1.5 && x2 > 1.5){
			return 1;
		}//13
		
		if(d4 < 0.2 && x1 < 1.5 && x2 > 1.5){
			return 2;
		}//14
		
		if(d4 < 0.2 && x1 > 1.5 && x2 < 1.5){
			return 2;
		}//15
		
		if(d4 < 0.2 && x1 < 1.5 && x2 < 1.5){
			return 1;
		}//16

		if(d5 > 0){
			return 3; //17
		} else if(d5 < 0){
			return 4; //18
		}else{
			return 3; //rare condition where d5 == 0
		}
	}
		
	public static void main(String[] args){
		//List for training set
		List<DataPoint> trainSet = new ArrayList<>();
		
		//List for testing set
		List<DataPoint> testSet = new ArrayList<>();
		
		Random rnd = new Random(1234); //stable seed
		
		int totalPoints = 8000;
		
		for(int i = 0; i<totalPoints; i++){
			double x1 = 2.0 * rnd.nextDouble();
			double x2 = 2.0 * rnd.nextDouble();
			
			int label = classify(x1, x2);
			
			DataPoint p = new DataPoint(x1, x2, label);
			
			//first 4000 -> training set
			if(i < 4000){
				trainSet.add(p);
			}else{
				testSet.add(p);
			}
		}
		
		System.out.println("Train size: " + trainSet.size());
		System.out.println("Test size: " + testSet.size());
		
		//Category counting for training set
		int c1_train = 0, c2_train = 0, c3_train = 0, c4_train = 0;
		
		for(DataPoint p : trainSet){
			if(p.label == 1) c1_train++;
			else if(p.label == 2) c2_train++;
			else if(p.label == 3) c3_train++;
			else if(p.label == 4) c4_train++;
		}
		
		//Category counting for test set
		int c1_test = 0, c2_test = 0, c3_test = 0, c4_test = 0;
		
		for(DataPoint p : testSet){
			if(p.label == 1) c1_test++;
			else if(p.label == 2) c2_test++;
			else if(p.label == 3) c3_test++;
			else if(p.label == 4) c4_test++;
		}
		
		System.out.println("\nTraining set:");
		System.out.println("C1: " + c1_train);
		System.out.println("C2: " + c2_train);
		System.out.println("C3: " + c3_train);
		System.out.println("C4: " + c4_train);
		
		System.out.println("\nTest set:");
		System.out.println("C1: " + c1_test);
		System.out.println("C2: " + c2_test);
		System.out.println("C3: " + c3_test);
		System.out.println("C4: " + c4_test);
		
		try{
			//Saving in csv files
			PrintWriter outTrain = new PrintWriter("train_T.csv");
			outTrain.println("x1,x2,label");
			
			for(DataPoint p : trainSet){
				outTrain.println(p.x1 + "," + p.x2 + "," + p.label);
			}
			outTrain.close();	
			
			PrintWriter outTest = new PrintWriter("test_T.csv");
			outTest.println("x1,x2,label");
			
			for(DataPoint p : testSet){
				outTest.println(p.x1 + "," + p.x2 + "," + p.label);
			}
			outTest.close();
			
			System.out.println("\nFiles train_t.csv and test_T.csv created");
		} catch (IOException e){
			System.out.println("Error during file writing: " + e.getMessage());
		}	
	}
}	
