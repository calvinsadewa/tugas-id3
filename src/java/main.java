/**
 * Created by calvin-pc on 9/25/2015.
 */

import weka.classifiers.trees.j48.*;
public class main {
    public static void main(String[] args) throws Exception{
        float cf = 0.5f;
        WekaAccessor wa = new WekaAccessor();
        wa.loadData("D:\\Program Files (x86)\\Weka-3-6\\data\\breast-cancer.arff");
        wa.supervisedResample();
        wa.classifier = new myJ48(cf);
        wa.classifier.buildClassifier(wa.data);
        double[] hasil = wa.test(wa.data);
        System.out.println(wa.evaluation.toSummaryString());
    }
}