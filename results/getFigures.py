import numpy as np
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle


restDir = ''
policies = ['Oracle', 'Default_conf', 'NeuralBandit', 'ClassMAB_ES', 'ClassMAB']


sim_train = list()
sim_eval= list()

legend = ['Oracle', 'Default Conf.', 'NeuralBandit', 'ClassMAB (ES)', 'ClassMAB']

for i in range(len(policies)):
    train = pickle.load(open(restDir+policies[i]+'_train.p', "rb"))
    eval = pickle.load(open(restDir+policies[i]+'_evaluation.p', "rb"))

    sim_train.append(train)
    sim_eval.append(eval)

lw_val = 1
ls_val_2 = ['-', '--', ':', '-', '--', ':']
ls_val = ['-', '-', '-', '-', '-', '-', '-', '-']
c_val = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 'k', '#9467bd', 'y']

# stagesPerEpoch = 200

# nHourPerDay = 24

nStagesPerEpoch = sim_train[0]['sim_params']['nStagesPerDay']
nSimulatedEpoch = sim_train[0]['sim_params']['nSimulatedDays']
nSimulatedEpoch_axis = 144

# TRAINING

xAxis = np.arange(0, nStagesPerEpoch*nSimulatedEpoch)
xAxisEpoch = np.arange(0, nSimulatedEpoch)

# plt.figure(1)
# for i in range(len(policies)):
#     ax = plt.plot(xAxisEpoch, sim_train[i]['consumptionEpoch']/1e3, ls=ls_val[i], lw=lw_val, label=legend[i], color=c_val[i])
# # plt.axis([0, 23, 1455, 1500])
# _, _, y1, y2 = plt.axis()
# # plt.axis((0, 23, y1, y2))
# plt.ylabel('Consumption (kW)')
# plt.xlabel('Epoch')
# plt.grid(True)
# plt.legend(loc="best")
# # plt.savefig(restDir+'consumption.eps', bbox_inches='tight')
# # plt.savefig(restDir+'consumption.pdf', bbox_inches='tight')


# plt.figure(2)
# for i in range(len(policies)):
#     plt.plot(xAxis, sim_train[i]['qosStage'], ls=ls_val[i], lw=lw_val, label=legend[i])
# plt.ylabel('QoS')
# plt.xlabel('Stage')
# plt.grid(True)
# plt.legend(loc="upper left")
# # plt.savefig(restDir+'QoS.eps', bbox_inches='tight')
# # plt.savefig(restDir+'QoS.pdf', bbox_inches='tight')

plt.figure(2)
for i in range(len(policies)):
    vProbSatifQoS = np.zeros(nSimulatedEpoch)
    for j in range(nSimulatedEpoch):
        vProbSatifQoS[j] = np.mean(sim_train[i]['qosStage'][(nStagesPerEpoch*j):(nStagesPerEpoch*(j+1))] > 0)
    plt.plot(xAxisEpoch, vProbSatifQoS, ls=ls_val[i], lw=lw_val, label=legend[i], color=c_val[i])
_, _, y1, y2 = plt.axis()
plt.axis((0, nSimulatedEpoch-1, y1, y2))
plt.ylabel('Probability of satifying QoS')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend(loc="best")

plt.figure(3)
for i in range(len(policies)):
    # plt.plot(xAxisEpoch, sim_train[i]['utilityEpoch'], ls=ls_val_2[i], lw=lw_val, label=legend[i], color=c_val[i])
    plt.semilogy(xAxisEpoch, sim_train[i]['utilityEpoch'], ls=ls_val[i], lw=lw_val, label=legend[i], color=c_val[i])
_, _, y1, y2 = plt.axis()
plt.axis((0, nSimulatedEpoch-1, y1, y2))
plt.ylabel('Cost Function Value')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend(loc="best")
# plt.savefig(restDir+'utility.eps', bbox_inches='tight')
# plt.savefig(restDir+'utility.pdf', bbox_inches='tight')

plt.figure(4)
for i in range(len(policies)):
    plt.plot(xAxisEpoch, sim_train[i]['regretEpoch'], ls=ls_val[i], lw=2, label=legend[i], color=c_val[i])
_, _, y1, y2 = plt.axis()
plt.axis((0, nSimulatedEpoch-1, y1, y2))
plt.ylabel('Regret')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend(loc="best")
# plt.savefig(restDir+'utility.eps', bbox_inches='tight')
# plt.savefig(restDir+'utility.pdf', bbox_inches='tight')


# plt.figure(5)
# for i in range(len(policies)):
#     plt.plot(xAxis, sim_train[i]['actionIndex'], ls='', marker=',',label=policies[i])
# plt.ylabel('Selected Actions')
# plt.xlabel('Stage')
# plt.grid(True)
# plt.legend(loc="upper left")
# # plt.savefig(restDir+'action_index.eps', bbox_inches='tight')
# # plt.savefig(restDir+'action_index.pdf', bbox_inches='tight')


# Evaluation

# xAxisEval = np.arange(0, nStagesPerEpoch)
nStagesEvaluated = 144
xAxisEval = np.arange(0, nStagesEvaluated)

plt.figure(6)
for i in range(len(policies)):
    plt.plot(xAxisEval, sim_eval[i]['consumptionStage'], ls=ls_val[i], lw=1.5, label=legend[i], color=c_val[i])
    ind = np.where(sim_eval[i]['qosStage'] < 0)[0]
    plt.plot(xAxisEval[ind], sim_eval[i]['consumptionStage'][ind], ls='', marker='x', color=c_val[i], markersize=11, markeredgewidth=1.5)
_, _, y1, y2 = plt.axis()
plt.axis((0, nSimulatedEpoch_axis-1, y1, y2))
plt.ylabel('Consumption (W)')
plt.xlabel('Stage')
plt.grid(True)
plt.legend(loc="best")
# plt.savefig(restDir+'utility.eps', bbox_inches='tight')
# plt.savefig(restDir+'utility.pdf', bbox_inches='tight')

# plt.figure(7)
# for i in range(len(policies)):
#     plt.plot(xAxisEval, sim_eval[i]['qosStage'], ls=ls_val[i], lw=lw_val, label=legend[i], color=c_val[i])
# plt.ylabel('QoS')
# plt.xlabel('Stage')
# plt.grid(True)
# plt.legend(loc="best")
# # plt.savefig(restDir+'utility.eps', bbox_inches='tight')
# # plt.savefig(restDir+'utility.pdf', bbox_inches='tight')

# plt.figure(8)
# for i in range(len(policies)):
#     plt.plot(xAxisEval, sim_eval[i]['utilityStage'], ls=ls_val[i], lw=lw_val, label=legend[i], color=c_val[i])
# plt.ylabel('Utility Function')
# plt.xlabel('Stage')
# plt.grid(True)
# plt.legend(loc="best")
# # plt.savefig(restDir+'utility.eps', bbox_inches='tight')
# # plt.savefig(restDir+'utility.pdf', bbox_inches='tight')

plt.figure(9)
for i in range(len(policies)):
    plt.plot(xAxisEval, sim_eval[i]['regretStage'], ls=ls_val[i], lw=2, label=legend[i], color=c_val[i])
_, _, y1, y2 = plt.axis()
plt.axis((0, nSimulatedEpoch_axis-1, y1, y2))
plt.ylabel('Regret')
plt.xlabel('Stage')
plt.grid(True)
plt.legend(loc="best")
# plt.savefig(restDir+'utility.eps', bbox_inches='tight')
# plt.savefig(restDir+'utility.pdf', bbox_inches='tight')

# plt.figure(10)
# for i in range(len(policies)):
#     plt.plot(xAxisEval, sim_eval[i]['actionIndex'], ls='', marker='.', label=legend[i], color=c_val[i])
# plt.ylabel('Selected Actions')
# plt.xlabel('Stage')
# plt.grid(True)
# plt.legend(loc="best")
# # plt.savefig(restDir+'utility.eps', bbox_inches='tight')
# # plt.savefig(restDir+'utility.pdf', bbox_inches='tight')


# plt.figure(3).gca().get_yaxis().get_major_formatter().set_powerlimits((0, 0))
plt.figure(9).gca().get_yaxis().get_major_formatter().set_powerlimits((0, 0))


plt.show()

segs = 60*10
fixedPolicy_index = 1
for i in range(len(policies)):
    print(policies[i])
    # print(np.sum(sim_eval[i]['consumptionStage'] * segs))
    print(1 - np.sum(sim_eval[i]['consumptionStage'] * segs) / np.sum(sim_eval[fixedPolicy_index]['consumptionStage'] * segs))
    print(np.mean(sim_eval[i]['qosStage'] > 0))


# print(policies[0])
# print(1-np.sum(consumptionList[0] * 3600) / np.sum(consumptionList[2] * 3600))
# print(policies[1])
# print(1-np.sum(consumptionList[1] * 3600) / np.sum(consumptionList[2] * 3600))
