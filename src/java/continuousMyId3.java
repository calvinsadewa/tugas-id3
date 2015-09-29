import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.filters.supervised.attribute.Discretize;

/**
 * Created by user on 28/09/2015.
 */
public class continuousMyId3 extends Classifier {

    FilteredClassifier m_root = null;

    public continuousMyId3() {
        this.m_root = new FilteredClassifier();
        m_root.setFilter(new Discretize());
        m_root.setClassifier(new myId3());
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        m_root.buildClassifier(data);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception{
        return m_root.classifyInstance(instance);
    }
}