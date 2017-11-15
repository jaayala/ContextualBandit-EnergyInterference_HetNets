import os.path
import numpy as np
import pickle
from sklearn.externals import joblib
from classification import neural_classif_usage as nn
from learnAlgs import neuralBandit as neuralband
from learnAlgs import Online_class_mab_ES as onlineCB_ES
from learnAlgs import Online_class_mab as onlineCB

evalAfterTraining = 1

nPicos = 6

nSimulatedDays = 1200
nStagesPerDay = 200
nHoursPerDay = 24
UEgenerationFactor = 10000
UEgenerationbias = 150

nABSvals = 8
ABSval = np.linspace(0, 7, nABSvals)
CREval = np.array([0, 6, 9, 12, 18])


nActivePicosVal = np.arange(0, (nPicos+1))
controlSpace = np.array(np.meshgrid(nActivePicosVal, ABSval, CREval)).T.reshape(-1, 3)
nControls = len(controlSpace[:, 0])

delta = 100

model_path = 'classification/models/'
model_file = 'models_con_qos600'

policies = ['Oracle', 'Default_conf', 'NeuralBandit', 'ClassMAB_ES', 'ClassMAB']

initP = nPicos  # Initialization
hoursVector = np.tile(np.arange(nStagesPerDay), (1, nSimulatedDays))[0]

QoSthres = 100000
QoSthresNN = 0

C = joblib.load(model_path+'regr_model.pkl')
Q = nn.Classification(model_path, model_file)
averageTrafficPerHour = np.array([0.025, 0.07, 0.085, 0.0964, 0.1, 0.105, 0.11, 0.115, 0.12, 0.1225, 0.125, 0.13, 0.14, 0.16, 0.155, 0.15, 0.125, 0.1, 0.075, 0.05, 0.03, 0.024, 0.024, 0.027])
averageTrafficPerHour = averageTrafficPerHour * UEgenerationFactor - UEgenerationbias
trafficPoly = np.polyfit(np.arange(nHoursPerDay), averageTrafficPerHour, 7)
axis = np.linspace(0, nHoursPerDay, nStagesPerDay)
averageTrafficPerStage = np.polyval(trafficPoly, axis)


initExploration = 20
epsilon_0 = 30
nb = neuralband.NeuralBandit(nPicos, ABSval, CREval, initExploration, epsilon_0)

initExploration = 20
epsilon_0 = 30
ocm_ES = onlineCB_ES.Online_class_mab(nPicos, ABSval, CREval, initExploration, epsilon_0)

initExploration = 20
epsilon_0 = 30
ocm = onlineCB.Online_class_mab(nPicos, ABSval, CREval, initExploration, epsilon_0)


algoList = list()
algoList.append('Oracle')
algoList.append('Def_conf')
algoList.append(nb)
algoList.append(ocm_ES)
algoList.append(ocm)

for p in range(len(policies)):

    print('Policy: '+policies[p])

    vConsumption = np.zeros((nSimulatedDays*nStagesPerDay))
    vQoS = np.zeros((nSimulatedDays*nStagesPerDay))
    vUtilityFunction = np.zeros((nSimulatedDays*nStagesPerDay))
    vActionIndex = np.zeros((nSimulatedDays*nStagesPerDay))

    vDayConsumption = np.zeros((nSimulatedDays))
    vDayUtility = np.zeros((nSimulatedDays))
    vDayRegret = np.zeros((nSimulatedDays))

    currentP = initP
    d = 0

    algorithm = algoList[p]

    if algorithm == 'Oracle':
        algConf = 'Oracle'
    elif algorithm == 'Def_conf':
        algConf = 'Def_conf'
    else:
        algConf = algorithm.getConf()

    for i in range(len(hoursVector)):
        k = hoursVector[i]

        traffic = np.random.poisson(averageTrafficPerStage[k])
        state = np.concatenate([np.array([traffic]), np.array([currentP])])

        inputData = {'k': k,
                     'state': state}

        state_controlSpace = np.append(np.tile(traffic, (nControls, 1)), controlSpace.copy(), axis=1)
        con_all = C.predict(state_controlSpace)
        qos_all = Q.getQoS(state_controlSpace).transpose()[0]
        utility_all = con_all + delta * np.maximum(np.zeros(nControls), QoSthresNN - qos_all)

        if algorithm == 'Oracle':
            index = np.argmin(utility_all)
            control = controlSpace[index, :]
        elif algorithm == 'Def_conf':
            control = np.array([6,0,0])
            index = 48
        else:
            control, index = algorithm.getControl(inputData)


        stateControlVector = np.concatenate([np.array([traffic]), control])
        currentConsumption = C.predict(np.expand_dims(stateControlVector, axis=0))
        currentQoS = Q.getQoS(np.expand_dims(stateControlVector, axis=0))
        utilityFunctionVal = currentConsumption + delta * np.max([0, QoSthresNN - currentQoS])
        regret = utilityFunctionVal - np.min(utility_all)


        inputData = {'k': k,
                     'index': index,
                     'state': state,
                     'control': control,
                     'utilityFunctionVal': utilityFunctionVal,
                     'currentConsumption': currentConsumption,
                     'currentQoS': currentQoS}

        if not isinstance(algorithm, str):
            algorithm.updateAlg(inputData)


        vConsumption[i] = currentConsumption
        vQoS[i] = currentQoS
        vUtilityFunction[i] = utilityFunctionVal
        vActionIndex[i] = index
        vDayConsumption[d] += currentConsumption
        vDayUtility[d] += utilityFunctionVal
        vDayRegret[d] += regret

        if k == nStagesPerDay-1:
            d += 1
            if np.mod(d, 10) == 0:
                print('Day '+str(d)+'/'+str(nSimulatedDays))

        currentP = control[0]

    sim_params = dict(nPicos=nPicos, nSimulatedDays=nSimulatedDays, nStagesPerDay=nStagesPerDay, delta=delta, policies=policies, algConf=algConf)
    sim_res = dict(sim_params=sim_params, consumptionStage=vConsumption, qosStage=vQoS, utilityStage=vUtilityFunction, actionIndex=vActionIndex, consumptionEpoch=vDayConsumption, utilityEpoch=vDayUtility, regretEpoch=np.cumsum(vDayRegret))

    restDir = 'results/'
    filename = policies[p]+'_train.p'
    if not os.path.exists(restDir):
        os.makedirs(restDir)
    pickle.dump(sim_res, open(restDir+filename, "wb"))
    print("Model saved in file: %s" % filename)


    # EVALUATION AFTER TRAINING: ONE DAY

    if evalAfterTraining:
        nStagesPerDayEval = 144  # Do not change
        axis = np.linspace(0, nHoursPerDay, nStagesPerDayEval)
        averageTrafficPerStageEval = np.polyval(trafficPoly, axis)
        currentP = nPicos

        vConsumptionEval = np.zeros(nStagesPerDayEval)
        vQoSEval = np.zeros(nStagesPerDayEval)
        vUtilityFunctionEval = np.zeros(nStagesPerDayEval)
        vActionIndexEval = np.zeros(nStagesPerDayEval)
        vRegretEval = np.zeros(nStagesPerDayEval)

        for k in range(len(averageTrafficPerStageEval)):

            traffic = np.random.poisson(averageTrafficPerStageEval[k])
            state = np.concatenate([np.array([traffic]), np.array([currentP])])

            inputData = {'k': k,
                         'state': state}

            state_controlSpace = np.append(np.tile(traffic, (nControls, 1)), controlSpace.copy(), axis=1)
            con_all = C.predict(state_controlSpace)
            qos_all = Q.getQoS(state_controlSpace).transpose()[0]
            utility_all = con_all + delta * np.maximum(np.zeros(nControls), QoSthresNN - qos_all)

            if algorithm == 'Oracle':
                index = np.argmin(utility_all)
                control = controlSpace[index, :]
            elif algorithm == 'Def_conf':
                control = np.array([6,0,0])
                index = 48
            else:
                control, index = algorithm.getControl(inputData)

            stateControlVector = np.concatenate([np.array([traffic]), control])
            currentConsumption = C.predict(np.expand_dims(stateControlVector, axis=0))
            currentQoS = Q.getQoS(np.expand_dims(stateControlVector, axis=0))
            utilityFunctionVal = currentConsumption + delta * np.max([0, QoSthresNN - currentQoS])
            regret = utilityFunctionVal - np.min(utility_all)


            vConsumptionEval[k] = currentConsumption
            vQoSEval[k] = currentQoS
            vUtilityFunctionEval[k] = utilityFunctionVal
            vActionIndexEval[k] = index
            vRegretEval[k] = regret


        sim_res = dict(sim_params=sim_params, consumptionStage=vConsumptionEval, qosStage=vQoSEval, utilityStage=vUtilityFunctionEval, actionIndex=vActionIndexEval, regretStage=np.cumsum(vRegretEval))

        restDir = 'results/'
        filename = policies[p]+'_evaluation.p'
        if not os.path.exists(restDir):
            os.makedirs(restDir)
        pickle.dump(sim_res, open(restDir+filename, "wb"))
        print("Model saved in file: %s" % filename)


Q.closeModel()
