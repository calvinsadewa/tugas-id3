import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.ADTree;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.FileInputStream;
import java.util.Random;

/**
 * Created by khaidzir on 25/09/2015.
 */
public class WekaAccessor {

    public Instances data;
    public Classifier classifier;
    Evaluation evaluation;

    public WekaAccessor() {
        data = null;
        classifier = null;
        evaluation = null;
    }

    public void loadData(String path) {
        DataSource source = null;
        Instances data = null;
        try {
            //source = new DataSource("C:\\Users\\user\\Desktop\\weka-3-6-13\\data\\weather.nominal.arff");
            source = new DataSource(path);
            data = source.getDataSet();
            data.setClassIndex(data.numAttributes()-1);
            System.out.println(data.classAttribute());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void removeAttributeAt(int position) {
        if (data == null) {
            return;
        }
        data.deleteAttributeAt(position);
    }

    public void buildNaiveBayesClassifier() {
        classifier = (Classifier)new NaiveBayes();
        try {
            classifier.buildClassifier(data);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void buildDTClassifier() {
        classifier = (Classifier)new ADTree();
        try {
            classifier.buildClassifier(data);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void crossValidation() {
        try {
            evaluation = new Evaluation(data);
            evaluation.crossValidateModel(classifier, data, 10, new Random(1));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void percentageSplit(double percent) {
        int trainSize = (int) Math.round(data.numInstances() * percent / 100);
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);
        try {
            classifier.buildClassifier(train);
            /*for (int i = 0; i < test.numInstances(); i++) {
                double pred = classifier.classifyInstance(test.instance(i));
                System.out.print("ID: " + test.instance(i));
                System.out.print(", actual: " + test.classAttribute().value((int) test.instance(i).classValue()));
                System.out.println(", predicted: " + test.classAttribute().value((int) pred));
            }*/
            evaluation = new Evaluation(train);
            evaluation.evaluateModel(classifier, test);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void loadModel(String filename) {
        try {
            classifier = (Classifier)SerializationHelper.read(new FileInputStream(filename));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void saveMode(String filename) {
        try {
            SerializationHelper.write(filename, classifier);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
