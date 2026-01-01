import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import java.io.PrintWriter;

public class PT_MLP{
    
    static final int D = 2; //number of inputs
    static final int K = 4; // number of categories
    static final int N_TRAIN = 4000;
    static final int N_TEST  = 4000;

    // nubmer of neurons in hidden layers
    static final int H1 = 30;
    static final int H2 = 30;
    static final int H3 = 30;

    //types of activation functions
    static final int ACT_LOGISTIC = 1;
    static final int ACT_TANH     = 2;
    static final int ACT_RELU     = 3;

    //Activation function
    static final int HIDDEN_ACT_1 = ACT_TANH;
    static final int HIDDEN_ACT_2 = ACT_TANH;
    static final int HIDDEN_ACT_3 = ACT_RELU;

    //learning rate
    static final double ETA = 0.01;

    //katofli diaforas sfalmatos metaxi epoxon gia termatismo
    static final double THRESHOLD = 0.0001;

    //minimum numbers of epochs
    static final int MIN_EPOCHS = 800;

    //Size of mini-batch L
    static final int BATCH_SIZE = N_TRAIN / 200;

    //Training inputs: N_TRAIN x D
    static double[][] trainX = new double[N_TRAIN][D];
    
    //Training targets: N_TRAIN x K
    static double[][] trainT = new double[N_TRAIN][K];

    //Test inputs: N_TEST x D
    static double[][] testX = new double[N_TEST][D];

    //Test targets: N_TEST x K
    static double[][] testT = new double[N_TEST][K];

    //Vari apo eisodo -> 1o krufo
    static double[][] W1 = new double[H1][D];
    static double[]   b1 = new double[H1];

    //1o -> 2o
    static double[][] W2 = new double[H2][H1];
    static double[]   b2 = new double[H2];

    //2o -> 3o
    static double[][] W3 = new double[H3][H2];
    static double[]   b3 = new double[H3];

    //3o -> out
    static double[][] W4 = new double[K][H3];
    static double[]   b4 = new double[K];

    static double[] h1 = new double[H1];
    static double[] h2 = new double[H2];
    static double[] h3 = new double[H3];

    //Out
    static double[] y = new double[K];

    //Gradients
    static double[][] gradW1 = new double[H1][D];
    static double[]   gradb1 = new double[H1];
    static double[][] gradW2 = new double[H2][H1];
    static double[]   gradb2 = new double[H2];
    static double[][] gradW3 = new double[H3][H2];
    static double[]   gradb3 = new double[H3];
    static double[][] gradW4 = new double[K][H3];
    static double[]   gradb4 = new double[K];

    //sintelestes sfalmatos se kathe epipedo
    static double[] delta1 = new double[H1];
    static double[] delta2 = new double[H2];
    static double[] delta3 = new double[H3];
    static double[] delta4 = new double[K];

    static void encodeOneHot(int label, double[] t){
        for(int k=0; k<K; k++){
            t[k] = 0.0;
        }
        t[label-1] = 1.0; 
    }

    static void loadDataset(String filename, boolean toTrain) throws IOException{
        BufferedReader br = new BufferedReader(new FileReader(filename));

        String line;
        int index = 0;

        //Ignore header
        line = br.readLine();

        while((line = br.readLine()) != null){
            String[] parts = line.split(",");

            double x1 = Double.parseDouble(parts[0].trim());
            double x2 = Double.parseDouble(parts[1].trim());
            int label = Integer.parseInt(parts[2].trim());

            if(toTrain){
                //filling training data
                trainX[index][0] = x1;
                trainX[index][1] = x2;

                encodeOneHot(label, trainT[index]);
            }else{
                //filling test data
                testX[index][0] = x1;
                testX[index][1] = x2;

                encodeOneHot(label, testT[index]);
            }
            index++;    
        }
        br.close();
    }
    //returns random value between (-1,1)
    static double randWeight(){
        return 2.0 * Math.random() - 1.0;
    }

    //random initialization of weights and polarizations
    static void initWeights(){
        //input -> 1st hidden
        for(int j=0; j<H1; j++){
            for(int i=0; i<D; i++){
                W1[j][i] = randWeight();
            }
            b1[j] = randWeight();
        }

        //1st -> 2nd hidden
        for(int j=0; j<H2; j++){
            for(int i=0; i<H1; i++){
                W2[j][i] = randWeight();
            }
            b2[j] = randWeight();
        }

        //2nd -> 3rd hidden
        for(int j=0; j<H3; j++){
            for(int i=0; i<H2; i++){
                W3[j][i] = randWeight();
            }
            b3[j] = randWeight();
        }

        //3rd -> out
        for(int k=0; k<K; k++){
            for(int j=0; j<H3; j++){
                W4[k][j] = randWeight();
            }
            b4[k] = randWeight();
        }
    }

    static double logistic(double u){
        return 1.0 / (1.0 + Math.exp(-u));
    }

    static double tanh(double u){
        return Math.tanh(u);
    }

    static double relu(double u){
        return (u>0.0) ? u : 0.0;
    }

    //choose activation function based on type
    static double activate(double u, int type){
        if(type == ACT_LOGISTIC) return logistic(u);
        if(type == ACT_TANH)     return tanh(u);
        if(type == ACT_RELU)     return relu(u);

        return u;
    }

    static double activationDerivative(double h, int type){
        if(type == ACT_LOGISTIC){
            return h * (1.0 - h);
        }
        if(type == ACT_TANH){
            return 1.0 - h * h;
        }
        if(type == ACT_RELU){
            return (h > 0.0) ? 1.0 : 0.0;
        }
        return 1.0;
    }

    static void forward(double[] x){
        //1st hidden layer
        // u1[j] = sum_i W1[j][i] * x[i] + b1[j]
        // h1[j] = f(u1[j])
        for(int j=0; j<H1; j++){
            double u = b1[j];

            for(int i=0; i<D; i++){
                u += W1[j][i] * x[i];
            }

            h1[j] = activate(u, HIDDEN_ACT_1);
        }

        for(int j=0; j<H2; j++){
            double u = b2[j];

            for(int i=0; i<H1; i++){
                u += W2[j][i] * h1[i];
            }

            h2[j] = activate(u, HIDDEN_ACT_2);
        }

        for(int j=0; j<H3; j++){
            double u = b3[j];

            for(int i=0; i<H2; i++){
                u += W3[j][i] * h2[i];
            }

            h3[j] = activate(u, HIDDEN_ACT_3);
        }

        for(int k=0; k<K; k++){
            double u = b4[k];

            for(int j=0; j<H3; j++){
                u += W4[k][j] * h3[j];
            }

            y[k] = logistic(u);
        }
    }

    static double backprop(double[] x, double[] t){

        forward(x);

        //computation of E error
        double E = 0.0;
        for(int k=0; k<K; k++){
            double e = t[k] - y[k];
            E += 0.5 * e * e;
        }

        //delta4[k]  = (y[k] - t[k]) * y[k] * (1 - y[k])
        for(int k=0; k<K; k++){
            double diff = y[k] - t[k];
            double dydU = y[k] * (1.0 - y[k]);
            delta4[k] = diff * dydU;
        }

        //delta3
        for(int j=0; j<H3; j++){
            double sum = 0.0;
            for(int k=0; k<K; k++){
                sum += delta4[k] * W4[k][j];
            }
            double dh = activationDerivative(h3[j], HIDDEN_ACT_3);
            delta3[j] = sum * dh;
        }

        //delta2
        for(int i=0; i<H2; i++){
            double sum = 0.0;
            for(int j=0; j<H3; j++){
                sum += delta3[j] * W3[j][i];
            }
            double dh = activationDerivative(h2[i], HIDDEN_ACT_2);
            delta2[i] = sum * dh;
        }

        //delta1
        for(int i=0; i<H1; i++){
            double sum = 0.0;
            for(int j=0; j<H2; j++){
                sum += delta2[j] * W2[j][i];
            }
            double dh = activationDerivative(h1[i], HIDDEN_ACT_1);
            delta1[i] = sum * dh;
        }

        //Accumulate gradients for output layer
        for(int k=0; k<K; k++){
            for(int j=0; j<H3; j++){
                gradW4[k][j] += delta4[k] * h3[j];
            }
            gradb4[k] += delta4[k];
        }

        //Accumulate gradients for 3rd hidden layer
        for(int j=0; j<H3; j++){
            for(int i=0; i<H2; i++){
                gradW3[j][i] += delta3[j] * h2[i];
            }
            gradb3[j] += delta3[j];
        }

        //Accumulate gradients for 2nd hidden layer
        for(int j=0; j<H2; j++){
            for(int i=0; i<H1; i++){
                gradW2[j][i] += delta2[j] * h1[i];
            }
            gradb2[j] += delta2[j];
        }

        //Accumulate gradients for 1st hidden layer
        for(int j=0; j<H1; j++){
            for(int i=0; i<D; i++){
                gradW1[j][i] += delta1[j] * x[i];
            }
            gradb1[j] += delta1[j];
        }

        return E;
    }

    //resets to zero every grad before every mini-batch
    static void zeroGradients(){
        for(int j=0; j<H1; j++){
            gradb1[j] = 0.0;
            for(int i=0; i<D; i++){
                gradW1[j][i] = 0.0;
            }
        }

        for(int j=0; j<H2; j++){
            gradb2[j] = 0.0;
            for(int i=0; i<H1; i++){
                gradW2[j][i] = 0.0;
            }
        }

        for(int j=0; j<H3; j++){
            gradb3[j] = 0.0;
            for(int i=0; i<H2; i++){
                gradW3[j][i] = 0.0;
            }
        }

        for(int k=0; k<K; k++){
            gradb4[k] = 0.0;
            for(int j=0; j<H3; j++){
                gradW4[k][j] = 0.0;
            }
        }
    }

    static void updateWeights(int batchSize){
        double scale = ETA / batchSize; //η/L

        for(int j=0; j<H1; j++){
            for(int i=0; i<D; i++){
                W1[j][i] -= scale * gradW1[j][i];
            b1[j] -= scale * gradb1[j];    
            }
        }

        for(int j=0; j<H2; j++){
            for(int i=0; i<H1; i++){
                W2[j][i] -= scale * gradW2[j][i];
            b2[j] -= scale * gradb2[j];    
            }
        }

        for(int j=0; j<H3; j++){
            for(int i=0; i<H2; i++){
                W3[j][i] -= scale * gradW3[j][i];
            b3[j] -= scale * gradb3[j];    
            }
        }

        for(int k=0; k<K; k++){
            for(int j=0; j<H3; j++){
                W4[k][j] -= scale * gradW4[k][j];
            b4[k] -= scale * gradb4[k];    
            }
        }
    }

    static void trainNetwork(){
        int N = trainX.length;
        int batchSize = BATCH_SIZE;
        int batches = N / batchSize;

        double prevError = 1e9;
        int epoch = 0;

        Random rnd = new Random(0);

        while(true){
             
            //randomizing samples
            for(int i=0; i<N; i++){
                int j = rnd.nextInt(N);
                double[] tmpX = trainX[i];
                trainX[i] = trainX[j];
                trainX[j] = tmpX;

                double[] tmpT = trainT[i];
                trainT[i] = trainT[j];
                trainT[j] = tmpT;
            }

            double epochError = 0.0;

            //Mini-batch loop
            for(int b=0; b<batches; b++){

                zeroGradients();

                int start = b * batchSize;
                int end = start + batchSize;

                for(int n = start; n<end; n++){
                    epochError += backprop(trainX[n], trainT[n]);
                }
                updateWeights(batchSize);
            }
            epoch++;

            System.out.println("Epoch " + epoch + " Error = " + epochError);

            if(epoch >= 800 && Math.abs(prevError - epochError) < THRESHOLD)
                break;

            prevError = epochError;
        }

        System.out.println("Training finished after " + epoch + " epochs.");
    }

    static int predictLabel(double[] x){
        forward(x);

        int bestK = 0;
        double bestVal = y[0];

        for(int k=1; k<K; k++){
            if(y[k] > bestVal){
                bestVal = y[k];
                bestK = k;
            }
        }
        return bestK + 1;
    }

    static void evaluateTestSet(){
        int correct = 0;
        int total = N_TEST;

        for(int n=0; n<N_TEST; n++){
            int pred = predictLabel(testX[n]);

            //true label from one-hot testT[n][]
            int trueLabel = -1;
            for(int k=0; k<K; k++){
                if(testT[n][k] == 1.0){
                    trueLabel = k + 1;
                    break;
                }
            }
            if(pred == trueLabel){
                correct++;
            }
        }
        double acc = 100.0 * correct / (double) total;
        System.out.println("Accuracy of test set: " + acc + " % (" + correct + "/" + total + ")");
    }

    static void writeTestResultsCSV(String filename){
        try(PrintWriter out = new PrintWriter(filename)){
            
            //header
            out.println("x1,x2,trueLabel,predLabel,correct");

            int correctCount = 0;

            for(int n=0; n<N_TEST; n++){
                double x1 = testX[n][0];
                double x2 = testX[n][1];

                int trueLabel = -1;
                for(int k=0; k<K; k++){
                    if(testT[n][k] == 1.0){
                        trueLabel = k+1;
                        break;
                    }
                }

                int predLabel = predictLabel(testX[n]);

                int correct = (predLabel == trueLabel) ? 1 : 0;
                if(correct == 1) correctCount++;

                out.println(x1 + "," + x2 + "," + trueLabel + "," + predLabel + "," + correct);
            }

            System.out.println("Test results written to " + filename);
            System.out.println("Correct: " + correctCount + " / " + N_TEST + " (" + (100.0 * correctCount / N_TEST) + " %)");

        }catch(Exception e){
            System.out.println("Error writing test results: " + e.getMessage());
        }
    }

    public static void main(String[] args){
        try{
            loadDataset("train_T.csv", true);
            loadDataset("test_T.csv", false);

            initWeights();
            //forward(trainX[0]);
            double E = backprop(trainX[0], trainT[0]);
            trainNetwork();
            evaluateTestSet();
            writeTestResultsCSV("test_results_T.csv");

            System.out.println("Loading train set and test set");
            System.out.println("N_TRAIN = " + N_TRAIN + ", N_TEST = " + N_TEST);
            System.out.println("D = " + D + ", K = " + K);
            System.out.println("H1 = " + H1 + ", H2 = " + H2 + ", H3 = " + H3);
            System.out.println("BATCH_SIZE = " + BATCH_SIZE);

            System.out.println("\nFirst training example:");
            System.out.print("x = [");
            System.out.print(trainX[0][0] + ", " + trainX[0][1] + "], t = [");
            for(int k=0; k<K; k++){
                System.out.print(trainT[0][k]);
                if(k<K-1) System.out.print(", ");
            }
            System.out.print("]");

            System.out.println("\nTest for random weights:");
            System.out.println("W1[0][0] = " +W1[0][0]);
            System.out.println("W2[0][0] = " +W2[0][0]);
            System.out.println("W3[0][0] = " +W3[0][0]);
            System.out.println("W4[0][0] = " +W4[0][0]);

            System.out.println("Output y for the first training example:");
            System.out.print("y = [");
            for(int k=0; k<K; k++){
                System.out.print(y[k]);
                if(k < K-1) System.out.print(", ");
            }
            System.out.print("]");

            System.out.println("\nError E for first training example: " + E);

        } catch(IOException e){
            System.out.println("Error during loading data: " + e.getMessage());
        }
    }
}