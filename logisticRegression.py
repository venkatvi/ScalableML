import os.path
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD
from math import log
from math import exp

def createOneHotDict(inputData):
    sampleDistinctFeats = (inputData
                            .flatMap(lambda x:x)
                            .distinct())
    sampleOHEDict = (sampleDistinctFeats
                    .zipWithIndex()
                    .collectAsMap())
    return sampleOHEDict

def oneHotEncoding(rawFeats, OHEDict, numOHEFeats):
    #return SparseVector(numOHEFeats, list(map((lambda x: (OHEDict[x], 1)), rawFeats)))
    encoding = []
    for i in range(len(rawFeats)):
        if rawFeats[i] in OHEDict:
            encoding.append((OHEDict[rawFeats[i]], 1))
    return SparseVector(numOHEFeats, encoding)

def parseOHEPoint(point, OHEDict, numOHEFeats):
    splitStr = point.split(',')
    label = splitStr[0]
    features = splitStr[1:]
    featureList = zip(range(len(features)), features);
    return LabeledPoint(label, oneHotEncoding(featureList, OHEDict, numOHEFeats))

def computeLogLoss(p,y):
    epsilon = 10e-12
    if y==1:
        if p==0:
            p = epsilon
        return -1*log(p)
    else:
        if p==1:
            return -1*log(epsilon)
        return -1*log(1-p)
def evaluateResults(model, data):
    predictions = data.map(lambda x : (getP(x.features, model.weights, model.intercept), x.label))
    return predictions.map(lambda x : computeLogLoss(x[0], x[1])).mean()

def getP(x, w, intercept):
    rawPrediction = x.dot(w) + intercept
    rawPrediction = min(rawPrediction, 20)
    rawPrediction = max(rawPrediction, -20)
    return 1/(1+exp(-rawPrediction))

def hashFunction(numBuckets, rawFeats):
    mapping = {}
    for ind, category in rawFeats:
        featureString = category + str(ind)
        mapping[featureString] = int(int(hashlib.md5(featureString).hexdigest(), 16) % numBuckets)
    sparseFeatures = defaultdict(float)
    for bucket in mapping.values():
        sparseFeatures[bucket] += 1.0
    return dict(sparseFeatures)

def parseHashPoint(point, numBuckets):
    splitStr = point.split(',')
    label = splitStr[0]
    features = splitStr[1:]
    featureList = zip(range(len(features)), features);
    return LabeledPoint(label, SparseVector(numBuckets, hashFunction(numBuckets, rawFeats)))

def computeSparsity(data, d, n):
    return data.map(lambda x : len(x.features.indices)).mean()/d

if '__name__' == '__main__':
    baseDir = os.path.join('data')
    inputPath = os.path.join('cs190', 'dac_sample.txt')
    fileName = os.path.join(baseDir, inputPath)

    if os.path.isfile(fileName):
        rawData = (sc
                   .textFile(fileName, 2)
                   .map(lambda x: x.replace('\t', ',')))  # work with either ',' or '\t' separated data
        weights = [.8, .1, .1]
        seed = 42
        # Use randomSplit with weights and seed
        rawTrainData, rawValidationData, rawTestData = rawData.randomSplit(weights, seed)
        # Cache the data
        rawTrainData.cache()
        rawValidationData.cache()
        rawTestData.cache()

        nTrain = rawTrainData.count()
        nVal = rawValidationData.count()
        nTest = rawTestData.count()

        #construct ctrOHEPointDict and num of features for CTROHE
        ctrOHEPoint = createOneHotDict(rawTrainData)
        numCtrOHEFeats = len(ctrOHEPoint.keys())

        OHETrainData = rawTrainData.map(lambda p : parseOHEPoint(point, ctrOHEPoint, numCtrOHEFeats))
        OHETrainData.cache()

        OHEValidationData = rawValidationData.map(lambda p : parseOHEPoint(point, ctrOHEPoint, numCtrOHEFeats))
        OHEValidationData.cache()

        #HyperParams for LOGISITICRegression
        # fixed hyperparameters
        numIters = 50
        stepSize = 10.
        regParam = 1e-6
        regType = 'l2'
        includeIntercept = True

        model0 = LogisticRegressionWithSGD.train(OHETrainData,numIters, stepSize, regParam=regParam, regType = regType, intercept = includeIntercept)

        baselineClass = OHETrainData.map(lambda x: int(x.label)).mean()
        logLossTrBaseline = OHETrainData.map(lambda x: computeLogLoss(baselineClass, x.label)).mean()
        predictionQuality = evaluateResults(model0, OHETrainData)
        # logLossTrBaseline vs. predictionQuality will give an estimate of the performance of the model

        baselineValClass = OHEValidationData.map(lambda x : int(x.label)).mean()
        logLossValBaseline = OHEValidationData.map(lambda x : computeLogLoss(baselineValClass, x.label)).mean()
        predictionValQuality = evaluateResults(model0, OHEValidationData)

        #Feature Hashing
        numBucketsCTR = 2**15
        hashTrainData = rawTrainData.map(lambda x : parseHashPoint(x, numBucketsCTR))
        hashTrainData.cache()
        hashValidationData = rawValidationData.map(lambda x : parseHashPoint(x, numBucketsCTR))
        hashValidationData.cache()
        hashTestData = rawTestData.map(lambda x : parseHashPoint(x, numBucketsCTR))
        hashTestData.cache()

        #Compute how sparse the OHE Features were compared to FeatureHashed features
        OHESparsity = computeSparsity(OHETrainData, numCtrOHEFeats, nTrain)
        hashSparsity = computeSparsity(hashTrainData, numBucketsCTR, nTrain)

        #Grid Search for HyperParams
        numIters = 500
        regType = 'l2'
        includeIntercept = True

        # Initialize variables using values from initial model training
        bestModel = None
        bestLogLoss = 1e10

        stepSizes = [1, 10]
        regParams = [1e-6, 1e-3]
        for stepSize in stepSizes:
            for regParam in regParams:
                model = (LogisticRegressionWithSGD
                         .train(hashTrainData, numIters, stepSize, regParam=regParam, regType=regType,
                                intercept=includeIntercept))
                logLossVa = evaluateResults(model, hashValidationData)
                print ('\tstepSize = {0:.1f}, regParam = {1:.0e}: logloss = {2:.3f}'
                       .format(stepSize, regParam, logLossVa))
                if (logLossVa < bestLogLoss):
                    bestModel = model
                    bestLogLoss = logLossVa


        #Evaluate test set
        baseLineTest = hashTestData.map(lambda x : int(x.label)).mean()
        logLossTest = hashTestData.map(lambda x: computeLogLoss(baseLineTest, x.label)).mean()
        predictionTestQuality = evaluateResults(bestModel, hashTestData)
