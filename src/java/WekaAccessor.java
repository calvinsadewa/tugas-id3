    import weka.classifiers.Classifier;
    import weka.classifiers.Evaluation;
    import weka.classifiers.bayes.NaiveBayes;
    import weka.classifiers.trees.ADTree;
    import weka.core.Instance;
    import weka.core.Instances;
    import weka.core.SerializationHelper;
    import weka.core.converters.ConverterUtils.DataSource;
    import weka.filters.supervised.instance.Resample;

    import java.io.FileInputStream;
    import java.util.Random;

    /**
     * Created by khaidzir on 25/09/2015.
     */
    public class WekaAccessor {

        public Instances trainData, testData;
        public Classifier classifier;
        Evaluation evaluation;

        public WekaAccessor() {
            trainData = null;
            classifier = null;
            evaluation = null;
        }


        public void loadData(String path) {
            DataSource source = null;
            trainData = null;
            testData = null;
            try {
                source = new DataSource(path);
                trainData = source.getDataSet();
                trainData.setClassIndex(trainData.numAttributes()-1);
                testData = trainData;
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        public void removeAttributeAt(int position) {
            if (trainData == null) {
                return;
            }
            trainData.deleteAttributeAt(position);
        }

        public void supervisedResample() {
            Resample resample = new Resample();
            resample.setRandomSeed((int)System.currentTimeMillis());
            try {
                resample.setInputFormat(trainData);
                trainData = Resample.useFilter(trainData, resample);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        public void unsupervisedResample() {
            weka.filters.unsupervised.instance.Resample resample = new weka.filters.unsupervised.instance.Resample();
            resample.setRandomSeed((int)System.currentTimeMillis());
            try {
                resample.setInputFormat(trainData);
                trainData = weka.filters.unsupervised.instance.Resample.useFilter(trainData, resample);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        public void buildNaiveBayesClassifier() {
            classifier = (Classifier)new NaiveBayes();
            try {
                classifier.buildClassifier(trainData);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        public void buildDTClassifier() {
            classifier = (Classifier)new ADTree();
            try {
                classifier.buildClassifier(trainData);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        public void buildMyID3Classifier() {
            classifier = (Classifier)new myId3();
            try {
                classifier.buildClassifier(trainData);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        public double[] test(Instances testSet) throws Exception {
            classifier.buildClassifier(trainData);
            evaluation = new Evaluation(trainData);
            return evaluation.evaluateModel(classifier, testSet);
        }

        public void crossValidation() throws Exception {
            classifier.buildClassifier(trainData);
            evaluation = new Evaluation(trainData);
            evaluation.crossValidateModel(classifier, testData, 10, new Random(1));
        }

        public void percentageSplit(double percent) throws Exception  {
            Instances dataset = new Instances(testData);
            dataset.randomize(new Random(1));
            int trainSize = (int) Math.round(dataset.numInstances() * percent / 100);
            int testSize = dataset.numInstances() - trainSize;
            Instances train = new Instances(dataset, 0, trainSize);
            Instances test = new Instances(dataset, trainSize, testSize);
            classifier.buildClassifier(train);
            evaluation = new Evaluation(train);
            evaluation.evaluateModel(classifier, test);
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

        public double classify(Instance instance) {
            double ret = -1.0f;
            try {
                ret = classifier.classifyInstance(instance);
            } catch (Exception e) {
                e.printStackTrace();
            }
            return ret;
        }

}
