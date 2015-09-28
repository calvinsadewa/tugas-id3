import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

import java.util.ArrayList;
import java.util.Enumeration;

/**
 * Modificated weka's ID3
 * the stop condition on building tree is when no more attribute can be split
 * rather than when max info gain of all attribute is zero
 * Also can classify instance that missing attribute by checking the distribution
 * of the best match tree,
 */
public class myId3
        extends Classifier
        implements TechnicalInformationHandler, Sourcable {

    /** for serialization */
    static final long serialVersionUID = -2693678647096322561L;

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
     * Returns a string describing the classifier.
     * @return a description suitable for the GUI.
     */
    public String globalInfo() {

        return  "Class for constructing an unpruned decision tree based on the ID3 "
                + "algorithm. Can only deal with nominal attributes. No missing values "
                + "allowed. Empty leaves may result in unclassified instances. For more "
                + "information see: \n\n"
                + getTechnicalInformation().toString();
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing
     * detailed information about the technical background of this class,
     * e.g., paper reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;

        result = new TechnicalInformation(Type.ARTICLE);
        result.setValue(Field.AUTHOR, "R. Quinlan");
        result.setValue(Field.YEAR, "1986");
        result.setValue(Field.TITLE, "Induction of decision trees");
        result.setValue(Field.JOURNAL, "Machine Learning");
        result.setValue(Field.VOLUME, "1");
        result.setValue(Field.NUMBER, "1");
        result.setValue(Field.PAGES, "81-106");

        return result;
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return      the capabilities of this classifier
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);

        // class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    /**
     * Builds Id3 decision tree classifier.
     *
     * @param data the training data
     * @exception Exception if classifier can't be built successfully
     */
    public void buildClassifier(Instances data) throws Exception {

        // can classifier handle the data?
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
     * @param data the training data
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

        // if data is "pure" (entrophy equal 0) or no attribute left
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
     * Computes class distribution for instance using decision tree.
     *
     * @param instance the instance for which distribution is to be computed
     * @return the class distribution for the given instance
     * @throws NoSupportForMissingValuesException if instance has missing values
     */
    public double[] distributionForInstance(Instance instance)
            throws NoSupportForMissingValuesException {

        if (m_Attribute == null) {
            return m_Distribution;
        } else if (instance.isMissing(m_Attribute)) {
            //if missing attribute supossed to used
            return m_Distribution;
        } else {
            return m_Successors[(int) instance.value(m_Attribute)].
                    distributionForInstance(instance);
        }
    }

    /**
     * Prints the decision tree using the private toString method from below.
     *
     * @return a textual description of the classifier
     */
    public String toString() {

        if ((m_Distribution == null) && (m_Successors == null)) {
            return "Id3: No model built yet.";
        }
        return "Id3\n\n" + toString(0);
    }

    /**
     * Computes information gain for an attribute.
     *
     * @param data the data for which info gain is to be computed
     * @param att the attribute
     * @return the information gain for the given attribute and data
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
     * @param data the data for which entropy is to be computed
     * @return the entropy of the data's class distribution
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
     * @param data the data which is to be split
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

    /**
     * Adds this tree recursively to the buffer.
     *
     * @param id          the unqiue id for the method
     * @param buffer      the buffer to add the source code to
     * @return            the last ID being used
     * @throws Exception  if something goes wrong
     */
    protected int toSource(int id, StringBuffer buffer) throws Exception {
        int                 result;
        int                 i;
        int                 newID;
        StringBuffer[]      subBuffers;

        buffer.append("\n");
        buffer.append("  protected static double node" + id + "(Object[] i) {\n");

        // leaf?
        if (m_Attribute == null) {
            result = id;
            if (Double.isNaN(m_ClassValue)) {
                buffer.append("    return Double.NaN;");
            } else {
                buffer.append("    return " + m_ClassValue + ";");
            }
            if (m_ClassAttribute != null) {
                buffer.append(" // " + m_ClassAttribute.value((int) m_ClassValue));
            }
            buffer.append("\n");
            buffer.append("  }\n");
        } else {
            buffer.append("    checkMissing(i, " + m_Attribute.index() + ");\n\n");
            buffer.append("    // " + m_Attribute.name() + "\n");

            // subtree calls
            subBuffers = new StringBuffer[m_Attribute.numValues()];
            newID = id;
            for (i = 0; i < m_Attribute.numValues(); i++) {
                newID++;

                buffer.append("    ");
                if (i > 0) {
                    buffer.append("else ");
                }
                buffer.append("if (((String) i[" + m_Attribute.index()
                        + "]).equals(\"" + m_Attribute.value(i) + "\"))\n");
                buffer.append("      return node" + newID + "(i);\n");

                subBuffers[i] = new StringBuffer();
                newID = m_Successors[i].toSource(newID, subBuffers[i]);
            }
            buffer.append("    else\n");
            buffer.append("      throw new IllegalArgumentException(\"Value '\" + i["
                    + m_Attribute.index() + "] + \"' is not allowed!\");\n");
            buffer.append("  }\n");

            // output subtree code
            for (i = 0; i < m_Attribute.numValues(); i++) {
                buffer.append(subBuffers[i].toString());
            }
            subBuffers = null;

            result = newID;
        }

        return result;
    }

    /**
     * Returns a string that describes the classifier as source. The
     * classifier will be contained in a class with the given name (there may
     * be auxiliary classes),
     * and will contain a method with the signature:
     * <pre><code>
     * public static double classify(Object[] i);
     * </code></pre>
     * where the array <code>i</code> contains elements that are either
     * Double, String, with missing values represented as null. The generated
     * code is public domain and comes with no warranty. <br/>
     * Note: works only if class attribute is the last attribute in the dataset.
     *
     * @param className the name that should be given to the source class.
     * @return the object source described by a string
     * @throws Exception if the source can't be computed
     */
    public String toSource(String className) throws Exception {
        StringBuffer        result;
        int                 id;

        result = new StringBuffer();

        result.append("class " + className + " {\n");
        result.append("  private static void checkMissing(Object[] i, int index) {\n");
        result.append("    if (i[index] == null)\n");
        result.append("      throw new IllegalArgumentException(\"Null values "
                + "are not allowed!\");\n");
        result.append("  }\n\n");
        result.append("  public static double classify(Object[] i) {\n");
        id = 0;
        result.append("    return node" + id + "(i);\n");
        result.append("  }\n");
        toSource(id, result);
        result.append("}\n");

        return result.toString();
    }

    /**
     * Returns the revision string.
     *
     * @return		the revision
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 6404 $");
    }

    /**
     * Main method.
     *
     * @param args the options for the classifier
     */
    public static void main(String[] args) {
        runClassifier(new myId3(), args);
    }
}


