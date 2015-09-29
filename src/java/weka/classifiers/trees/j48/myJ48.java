package weka.classifiers.trees.j48;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by user on 28/09/2015.
 */
public class myJ48 extends Classifier{
    myC45PruneableClassifierTree m_root = null;
    float crit_val = 0;

    public myJ48(float critical_value) {
        crit_val = critical_value;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        m_root = new myC45PruneableClassifierTree(new C45ModelSelection(0,data),true,crit_val,true,true);
        m_root.buildClassifier(data);
    }

    public double classifyInstance(Instance instance) throws Exception {
        return m_root.classifyInstance(instance);
    }
}
