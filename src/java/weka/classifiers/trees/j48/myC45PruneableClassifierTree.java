package weka.classifiers.trees.j48;

import weka.core.*;
import weka.core.Capabilities.Capability;

import java.util.Enumeration;

/**
 * Class for handling a tree structure that can
 * be pruned using C4.5 procedures.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 8986 $
 */

public class myC45PruneableClassifierTree
        extends ClassifierTree {

    /** for serialization */
    static final long serialVersionUID = -4813820170260388194L;

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
     * Returns default capabilities of the classifier tree.
     *
     * @return      the capabilities of this classifier tree
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.DATE_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);

        // class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
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
        collapse();
        if (m_pruneTheTree) {
            prune(data);
        }
        if (m_cleanup) {
            cleanup(new Instances(data, 0));
        }
    }

    /**
     * Collapses a tree to a node if training error doesn't increase.
     */
    public final void collapse(){

        double errorsOfSubtree;
        double errorsOfTree;
        int i;

        if (!m_isLeaf){
            errorsOfSubtree = getTrainingErrors();
            errorsOfTree = localModel().distribution().numIncorrect();
            if (errorsOfSubtree >= errorsOfTree-1E-3){

                // Free adjacent trees
                m_sons = null;
                m_isLeaf = true;

                // Get NoSplit Model for tree.
                m_localModel = new NoSplit(localModel().distribution());
            }else
                for (i=0;i<m_sons.length;i++)
                    son(i).collapse();
        }
    }

    /**
     * Prunes a tree using C4.5's pruning procedure.
     *
     * @throws Exception if something goes wrong
     */
    public boolean prune(Instances data) throws Exception {
        int i;
        double infoGain = computeInfoGain(data);

        if (infoGain < m_CF) {
            boolean pruneable = true;
            //Check child
            // Prune all subtrees.
            Instances[] datas = m_localModel.split(data);
            for (i=0;i<m_sons.length;i++)
                pruneable = pruneable && son(i).prune(datas[i]);

            if (pruneable == true) {
                m_isLeaf = true;
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
     * Computes estimated errors for tree.
     *
     * @return the estimated errors
     */
    private double getEstimatedErrors(){

        double errors = 0;
        int i;

        if (m_isLeaf)
            return getEstimatedErrorsForDistribution(localModel().distribution());
        else{
            for (i=0;i<m_sons.length;i++)
                errors = errors+son(i).getEstimatedErrors();
            return errors;
        }
    }

    /**
     * Computes estimated errors for one branch.
     *
     * @param data the data to work with
     * @return the estimated errors
     * @throws Exception if something goes wrong
     */
    private double getEstimatedErrorsForBranch(Instances data)
            throws Exception {

        Instances [] localInstances;
        double errors = 0;
        int i;

        if (m_isLeaf)
            return getEstimatedErrorsForDistribution(new Distribution(data));
        else{
            Distribution savedDist = localModel().m_distribution;
            localModel().resetDistribution(data);
            localInstances = (Instances[])localModel().split(data);
            localModel().m_distribution = savedDist;
            for (i=0;i<m_sons.length;i++)
                errors = errors+
                        son(i).getEstimatedErrorsForBranch(localInstances[i]);
            return errors;
        }
    }

    /**
     * Computes estimated errors for leaf.
     *
     * @param theDistribution the distribution to use
     * @return the estimated errors
     */
    private double getEstimatedErrorsForDistribution(Distribution
                                                             theDistribution){

        if (Utils.eq(theDistribution.total(),0))
            return 0;
        else
            return theDistribution.numIncorrect()+
                    Stats.addErrs(theDistribution.total(),
                            theDistribution.numIncorrect(), m_CF);
    }

    /**
     * Computes errors of tree on training data.
     *
     * @return the training errors
     */
    private double getTrainingErrors(){

        double errors = 0;
        int i;

        if (m_isLeaf)
            return localModel().distribution().numIncorrect();
        else{
            for (i=0;i<m_sons.length;i++)
                errors = errors+son(i).getTrainingErrors();
            return errors;
        }
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
     * Computes new distributions of instances for nodes
     * in tree.
     *
     * @param data the data to compute the distributions for
     * @throws Exception if something goes wrong
     */
    private void newDistribution(Instances data) throws Exception {

        Instances [] localInstances;

        localModel().resetDistribution(data);
        m_train = data;
        if (!m_isLeaf){
            localInstances =
                    (Instances [])localModel().split(data);
            for (int i = 0; i < m_sons.length; i++)
                son(i).newDistribution(localInstances[i]);
        } else {

            // Check whether there are some instances at the leaf now!
            if (!Utils.eq(data.sumOfWeights(), 0)) {
                m_isEmpty = false;
            }
        }
    }

    /**
     * Method just exists to make program easier to read.
     */
    private myC45PruneableClassifierTree son(int index){

        return (myC45PruneableClassifierTree)m_sons[index];
    }

    /**
     * Returns the revision string.
     *
     * @return		the revision
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 8986 $");
    }

    /**
     * Computes information gain for an attribute.
     *
     * @param data the data for which info gain is to be computed
     * @return the information gain for the given attribute and data
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
