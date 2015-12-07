import os.path
import numpy as np
from pyspark.mlib.regression import LabeledPoint
from pyspark.mlib.linalg import DenseVector
from pyspark.mlib.regression import LinearRegressioWithSGD

def parsePoint(line):
    values = line.split(' ')
    label = values[0]
    features = values[1:]
    return LabeledPoint(label, features)

def gradientSummand(weights, lp):
    predictedVal = weights.dot(lp.features);
    summand = (predictedVal - lp.label)*lp.features
    return summand

def getLabeledPrediction(weights, observation):
    label = observation.label
    prediction = weights.dot(observation.features)
    return (label, prediction)

def calcRMSE(labelAndPred):
    return sqrt((labelAndPred[0]-labelAndPred[1])**2)

def linearRegressionGradientDescent(trainData, numIters):
    n = trainData.count()
    d = len(trainData.take(1)[0].features)
    w = np.zeros(d)
    alpha = 1.0
    errorTrain = np.zeros(numIters)

    for i in range(numIters):
        labelsAndPredsTrain = trainData.map(lambda lp: getLabeledPrediction(w, lp))
        errorTrain[i] = calcRMSE(labelsAndPredsTrain)
        gradient = trainData.map(lambda lp: gradientSummand(w, lp)).sum()
        alpha_i = alpha/(n*np.sqrt(i+1))
        w -= gradient*alpha_i

    return w, errorTrain
if __name__ == '__main__':
    sc = SparkContext()

    baseDir = os.path.join('data')
    inputPath = os.path.join('cs190', 'millionsong.txt')
    fileName = os.path.join(basePath, inputPath)

    numPartitions = 2
    rawData = sc.textFile(fileName, numPartitions)

    # getData parsed into LabeledPoints
    parsedDataInit = rawData.map(parsePoint)
    onlyLabels = parsedDataInit.map(lambda x : int(x.label))
    minYear = onlyLabels.map(lambda x,y: min(x,y))
    maxear = onlyLabels.map(lambda x,y: max(x,y))

    #normalize labels
    parsedData = parsedDataInit.map(lambda x : LabeledPoint((int(x.label) - minYear), x.features))

    #split data
    weights = [0.8, 0.1, 0.1]
    seed = 42
    parsedTrainData, parsedValData, parsedTestData = parsedData.randomSplit(weights, seed)

    parsedTrainData.cache()
    parsedValData.cache()
    parsedTestData.cache()

    #deploy inhouse impl of LinearRegression GradDescent on toy sample set
    examplesN = 10
    examplesD = 3
    exampleData = (sc
                    .parallelize(parsedTrainData.tak(exampleN))
                    .map(lambda lp: LabeledPoint(lp.label, lp.features[0:exampleD])))
    exampleNumIters = 5
    exampleWts, exampleErrorTrain = linearRegressionGradientDescent(exampleData, exampleNumIters)

    # deploy inhouse impl on allData
    numIters = 50
    wt0, error0 = linearRegressionGradientDescent(parsedTrainData, numIters)
    #get predictions on validation set
    labelsAndPredsVal = parsedValData.map(lambda lp: getLabeledPrediction(wt0, lp))
    rmsePred = calcRMSE(labelsAndPredsVal)


    # deploy  linearRegressionSGD algorithm of mllib
    numIters = 500
    alpha = 1
    miniBatchFrac = 1
    reg = 1e-1
    regType = 'l2'
    useIntercept = True

    firstModel = LinearRegressioWithSGD.train(parsedTrainData, numIters, alpha, miniBatchFrac, regParam = reg, regType = regType, intercept = useIntercept)
    labelsAndPredValFirstModel = parsedValData.map(lambda lp: (x.label, firstModel.predict(x.features)))
    rmsePredFirstModel = calcRMSE(labelsAndPredValFirstModel)

    # find best regularization param val via grid search
    bestRMSE=rmsePredFirstModel
    bestRegParam = reg
    bestModel = firstModel
    for reg in [1e-10, 1e-05, 1]:
        model = LinearRegressioWithSGD.train(parsedTrainData, numIters, alpha, miniBatchFrac, regParam = reg, regType = regType, intercept = useIntercept)
        labelsAndPredValModel = parsedValData.map(lambda lp: (x.label, model.predict(x.features)))
        rmsePredModel = calcRMSE(labelsAndPredValModel)
        if rmsePredModel < rmsePredFirstModel:
            bestRMSE = rmsePredModel
            bestRegParam = reg
            bestModel = model

    rmseValGrid = bestRMSE

    # fixing lambda, choose different learning rates / step size and number of iterations
    reg = bestRegParam
    modelRMSEs = []
    for alpha in [1e-5, 10]:
        for numIters in [5,100]:
            model = LinearRegressioWithSGD.train(parsedTrainData, numIters, alpha, miniBatchFrac, regParam = reg, regType = regType, intercept = useIntercept)
            labelsAndPredValModel = parsedValData.map(lambda lp: (x.label, model.predict(x.features)))
            rmsePredModel = calcRMSE(labelsAndPredValModel)
            modelRMSEs.append(rmseVal)

    # Add two way interaction between features
