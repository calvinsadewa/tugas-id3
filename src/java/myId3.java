import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.Enumeration;

/**
 * Modified weka's ID3
 * the stop condition on building tree is when no more attribute can be split or the enthropy of trainData is 0
 * rather than when max info gain of all attribute is zero as in weka's
 * Also can classify instance that missing attribute by checking the distribution
 * of the best match tree,
 */
public class myId3
        extends Classifier {

    /** The node's successors. */
    private myId3[] m_Successors;

    /** Attribute used for splitting. */
    private Attribute m_Attribute;

    /** Class value if node is leaf. */
    private double m_ClassValue;

    /** Class distribution if node is leaf. */
    private double[] m_Distribution;

    /** Class attribute of dataset. */
    private Attribute m_ClassAttribute;

    /**
     * Builds Id3 decision tree classifier.
     *
     * @param data the training trainData
     * @exception Exception if classifier can't be built successfully
     */
    public void buildClassifier(Instances data) throws Exception {

        // can classifier handle the trainData?
        getCapabilities().testWithFail(data);

        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();

        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        for (int i = 0; i < data.numAttributes(); i++) {
            if (i != data.classIndex()) attributes.add(data.attribute(i));
        }
        makeTree(data,attributes,Instance.missingValue(),data.classAttribute());
    }

    /**
     * Method for building an Id3 tree.
     *
     * @param data the training trainData
     * @param attributes the list of attribute that can be selected to make tree
     * @param parentClassValue the parent class value
     * @param classAttribute the attribute to be classified
     * @exception Exception if decision tree can't be built successfully
     */
    private void makeTree(Instances data, ArrayList<Attribute> attributes,
                          double parentClassValue, Attribute classAttribute) throws Exception {

        m_ClassAttribute = classAttribute;

        // Check if no instances have reached this node.
        if (data.numInstances() == 0) {
            m_Attribute = null;
            m_ClassValue = parentClassValue;
            m_Distribution = new double[data.numClasses()];
            return;
        }

        m_Distribution = new double[data.numClasses()];
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            m_Distribution[(int) inst.classValue()]++;
        }
        m_ClassValue = Utils.maxIndex(m_Distribution);

        // if trainData is "pure" (entrophy equal 0) or no attribute left
        if (m_Distribution[Utils.maxIndex(m_Distribution)] == data.numInstances()
                || attributes.size() == 0 ){
            Utils.normalize(m_Distribution);
            m_Attribute = null;
            return;
        };

        Utils.normalize(m_Distribution);

        // Compute attribute with maximum information gain.
        double[] infoGains = new double[attributes.size()];
        for (int i = 0; i < attributes.size(); i++) {
            Attribute att = attributes.get(i);
            infoGains[i] = computeInfoGain(data, att);
        }
        m_Attribute = attributes.get(Utils.maxIndex(infoGains));

        Instances[] splitData = splitData(data, m_Attribute);
        m_Successors = new myId3[m_Attribute.numValues()];

        ArrayList<Attribute> newAttributes = new ArrayList<Attribute>(attributes);
        newAttributes.remove(m_Attribute);
        for (int j = 0; j < m_Attribute.numValues(); j++) {
            m_Successors[j] = new myId3();
            m_Successors[j].makeTree(splitData[j],newAttributes, m_ClassValue,classAttribute);
        }
    }

    /**
     * Classifies a given test instance using the decision tree.
     *
     * @param instance the instance to be classified
     * @return the classification
     * @throws NoSupportForMissingValuesException if instance has missing values
     */
    public double classifyInstance(Instance instance){

        //if leaf
        if (m_Attribute == null) {
            return m_ClassValue;
        }
        else if (instance.isMissing(m_Attribute)) {
            //if missing attribute supossed to used
            return m_ClassValue;
        } else {
            return m_Successors[(int) instance.value(m_Attribute)].
                    classifyInstance(instance);
        }
    }

    /**
     * Computes information gain for an attribute.
     *
     * @param data the trainData for which info gain is to be computed
     * @param att the attribute
     * @return the information gain for the given attribute and trainData
     * @throws Exception if computation fails
     */
    private double computeInfoGain(Instances data, Attribute att)
            throws Exception {

        double infoGain = computeEntropy(data);
        Instances[] splitData = splitData(data, att);
        for (int j = 0; j < att.numValues(); j++) {
            if (splitData[j].numInstances() > 0) {
                infoGain -= ((double) splitData[j].numInstances() /
                        (double) data.numInstances()) *
                        computeEntropy(splitData[j]);
            }
        }
        return infoGain;
    }

    /**
     * Computes the entropy of a dataset.
     *
     * @param data the trainData for which entropy is to be computed
     * @return the entropy of the trainData's class distribution
     * @throws Exception if computation fails
     */
    private double computeEntropy(Instances data) throws Exception {

        double [] classCounts = new double[data.numClasses()];
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            classCounts[(int) inst.classValue()]++;
        }
        double entropy = 0;
        for (int j = 0; j < data.numClasses(); j++) {
            if (classCounts[j] > 0) {
                entropy -= classCounts[j] * Utils.log2(classCounts[j]);
            }
        }
        entropy /= (double) data.numInstances();
        return entropy + Utils.log2(data.numInstances());
    }

    /**
     * Splits a dataset according to the values of a nominal attribute.
     *
     * @param data the trainData which is to be split
     * @param att the attribute to be used for splitting
     * @return the sets of instances produced by the split
     */
    private Instances[] splitData(Instances data, Attribute att) {

        Instances[] splitData = new Instances[att.numValues()];
        for (int j = 0; j < att.numValues(); j++) {
            splitData[j] = new Instances(data, data.numInstances());
        }
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            splitData[(int) inst.value(att)].add(inst);
        }
        for (int i = 0; i < splitData.length; i++) {
            splitData[i].compactify();
        }
        return splitData;
    }

    /**
     * Outputs a tree at a certain level.
     *
     * @param level the level at which the tree is to be printed
     * @return the tree as string at the given level
     */
    private String toString(int level) {

        StringBuffer text = new StringBuffer();

        if (m_Attribute == null) {
            if (Instance.isMissingValue(m_ClassValue)) {
                text.append(": null");
            } else {
                text.append(": " + m_ClassAttribute.value((int) m_ClassValue));
            }
        } else {
            for (int j = 0; j < m_Attribute.numValues(); j++) {
                text.append("\n");
                for (int i = 0; i < level; i++) {
                    text.append("|  ");
                }
                text.append(m_Attribute.name() + " = " + m_Attribute.value(j));
                text.append(m_Successors[j].toString(level + 1));
            }
        }
        return text.toString();
    }
}


