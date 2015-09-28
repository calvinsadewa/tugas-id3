/**
 * Created by calvin-pc on 9/25/2015.
 */

import weka.classifiers.trees.j48.*;
public class main {
    public static void main(String[] args) throws Exception{
        float cf = 0.5f;
        WekaAccessor wa = new WekaAccessor();
        wa.loadData("C:\\Users\\user\\Desktop\\weka-3-6-13\\data\\weather.numeric.arff");
        wa.supervisedResample();
        wa.classifier = new continuousMyId3();
        wa.classifier.buildClassifier(wa.data);
        double[] hasil = wa.test(wa.data);
        System.out.println(wa.evaluation.toSummaryString());
    }
}