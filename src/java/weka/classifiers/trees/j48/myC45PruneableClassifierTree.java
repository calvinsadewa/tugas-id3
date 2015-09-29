package weka.classifiers.trees.j48;

import weka.core.*;
import weka.core.Capabilities.Capability;

import java.util.Enumeration;

/**
 * Modified weka's C45PruneableClassifierTree
 *
 * Use critical value factor pruning rather than reduced error pruning
 */

public class myC45PruneableClassifierTree
        extends ClassifierTree {

    /** True if the tree is to be pruned. */
    boolean m_pruneTheTree = false;

    /** The critical value for pruning. */
    float m_CF = 0.25f;

    /** Is subtree raising to be performed? */
    boolean m_subtreeRaising = true;

    /** Cleanup after the tree has been built. */
    boolean m_cleanup = true;

    /**
     * Constructor for pruneable tree structure. Stores reference
     * to associated training data at each node.
     *
     * @param toSelectLocModel selection method for local splitting model
     * @param pruneTree true if the tree is to be pruned
     * @param cf the confidence factor for pruning
     * @param raiseTree
     * @param cleanup
     * @throws Exception if something goes wrong
     */
    public myC45PruneableClassifierTree(ModelSelection toSelectLocModel,
                                      boolean pruneTree,float cf,
                                      boolean raiseTree,
                                      boolean cleanup)
            throws Exception {

        super(toSelectLocModel);

        m_pruneTheTree = pruneTree;
        m_CF = cf;
        m_subtreeRaising = raiseTree;
        m_cleanup = cleanup;
    }

    /**
     * Method for building a pruneable classifier tree.
     *
     * @param data the data for building the tree
     * @throws Exception if something goes wrong
     */
    public void buildClassifier(Instances data) throws Exception {

        // can classifier tree handle the data?
        getCapabilities().testWithFail(data);

        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();

        buildTree(data, m_subtreeRaising || !m_cleanup);
        if (m_pruneTheTree) {
            prune(data);
        }
        if (m_cleanup) {
            cleanup(new Instances(data, 0));
        }
    }

    /**
     * Prunes a tree using critical value factor as in (Mingers, 1987).
     * The critical value used is info gain.
     *
     * @throws Exception if something goes wrong
     */
    public boolean prune(Instances data) throws Exception {
        int i;
        double infoGain = computeInfoGain(data);

        // If critical value do not reach threshold
        if (infoGain < m_CF) {
            boolean pruneable = true;
            //Check child
            // Prune all subtrees.
            Instances[] datas = m_localModel.split(data);
            for (i=0;i<m_sons.length;i++)
                pruneable = pruneable && son(i).prune(datas[i]);

            if (pruneable == true) {
                m_isLeaf = true;
                m_sons = null;
                return true;
            }
        }
        return false;
    }

    /**
     * Returns a newly created tree.
     *
     * @param data the data to work with
     * @return the new tree
     * @throws Exception if something goes wrong
     */
    protected ClassifierTree getNewTree(Instances data) throws Exception {

        myC45PruneableClassifierTree newTree =
                new myC45PruneableClassifierTree(m_toSelectModel, m_pruneTheTree, m_CF,
                        m_subtreeRaising, m_cleanup);
        newTree.buildTree((Instances)data, m_subtreeRaising || !m_cleanup);

        return newTree;
    }

    /**
     * Method just exists to make program easier to read.
     *
     * @return the local split model
     */
    private ClassifierSplitModel localModel(){

        return (ClassifierSplitModel)m_localModel;
    }

    /**
     * Method just exists to make program easier to read.
     */
    private myC45PruneableClassifierTree son(int index){

        return (myC45PruneableClassifierTree)m_sons[index];
    }

    /**
     * Computes information gain for current tree
     *
     * @param data the data for which info gain is to be computed
     * @return the information gain for the current tree and data
     * @throws Exception if computation fails
     */
    private double computeInfoGain(Instances data)
            throws Exception {

        double infoGain = computeEntropy(data);
        Instances[] splitData = m_localModel.split(data);
        for (int j = 0; j < m_localModel.numSubsets(); j++) {
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
}
