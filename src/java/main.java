/**
 * Created by calvin-pc on 9/25/2015.
 */

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.j48.*;
import weka.core.Instances;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class main {
    public static void main(String[] args){
        String pathFolderDataset = "C:\\Users\\user\\Desktop\\dataSet\\";
        float cf = 0.5f;
        ArrayList<Classifier> classifiers = new ArrayList<>();
        ArrayList<String> classifierNames = new ArrayList<>();
        ArrayList<String> datas = new ArrayList<>();
        ArrayList<String> dataNames = new ArrayList<>();

        // Tambahkan klasifier
        classifiers.add(new NaiveBayes());
        classifiers.add(new Id3());
        classifiers.add(new J48());
        classifiers.add(new continuousMyId3());
        classifiers.add(new myJ48(cf));

        // Tambahkan nama klasifier
        classifierNames.add("Naive Bayes");
        classifierNames.add("Id3");
        classifierNames.add("J48");
        classifierNames.add("MyID3");
        classifierNames.add("MyJ48");

        // Tambahkan dataset
        datas.add(pathFolderDataset + "iris.arff");
        datas.add(pathFolderDataset + "iris.missing.arff");
        datas.add(pathFolderDataset + "iris.2D.arff");
        datas.add(pathFolderDataset + "iris.2D.missing.arff");
        datas.add(pathFolderDataset + "weather.nominal.arff");
        datas.add(pathFolderDataset + "weather.nominal.missing.arff");
        datas.add(pathFolderDataset + "weather.numeric.arff");
        datas.add(pathFolderDataset + "weather.numeric.missing.arff");

        // Tambahkan nama dataset
        dataNames.add("Iris");
        dataNames.add("Iris Missing");
        dataNames.add("Iris 2D");
        dataNames.add("Iris 2D Missing");
        dataNames.add("Weather Nominal");
        dataNames.add("Weather Nominal Missing");
        dataNames.add("Weather Numeric");
        dataNames.add("Weather Numeric Missing");

        String hasil = "";
        WekaAccessor wa = new WekaAccessor();

        for (int j = 0; j < datas.size(); j++) {
            hasil += "=======================================================================\n";
            hasil += "                         DATA SET : ";
            hasil += dataNames.get(j) + "\n";
            hasil += "=======================================================================\n";

            wa.loadData(datas.get(j));
            Instances dataset = new Instances(wa.trainData);

            String summary = "";
            for (int i = 0; i < classifiers.size(); i++) {
                //WekaAccessor wa = new WekaAccessor();
                //wa.loadData(datas.get(j));
                //wa.supervisedResample();
                wa.trainData = new Instances(dataset);
                wa.testData = new Instances(dataset);
                wa.classifier = classifiers.get(i);
                summary = "-----------------------------------------------------------------------\n";
                summary += "                         CLASSIFIER : " + classifierNames.get(i) + "\n";
                summary += "-----------------------------------------------------------------------\n";

                for (int k=0; k<3; k++)
                    try {
                        switch (k) {
                            case 0:
                                summary += "TEST EVALUATION : Training Set";
                                wa.test(wa.trainData);
                                summary += wa.evaluation.toSummaryString() +
                                        System.lineSeparator();
                                break;
                            case 1:
                                summary += "TEST EVALUATION : Percentage split 70%";
                                wa.percentageSplit(70);
                                summary += wa.evaluation.toSummaryString() +
                                        System.lineSeparator();
                                break;
                            case 2:
                                summary += "TEST EVALUATION : Cross Validation";
                                wa.crossValidation();
                                summary += wa.evaluation.toSummaryString() +
                                        System.lineSeparator();
                                break;
                        }
                    } catch (Exception e) {
                        summary += "\n" + e.getMessage() + "\n\n";
                    }

                hasil += summary;
            }

            hasil += "=======================================================================\n\n\n\n\n";
        }
        saveToFile(hasil, "C:\\Users\\user\\Desktop\\hasil.txt");
    }

    public static void saveToFile(String data, String filename) {
        try {
            File file = new File(filename);
            if (!file.exists()) {
                file.createNewFile();
            }
            FileWriter fw = new FileWriter(file.getAbsoluteFile());
            BufferedWriter bw = new BufferedWriter(fw);
            bw.write(data);
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}