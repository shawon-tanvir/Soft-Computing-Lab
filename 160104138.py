import pandas as pd
import numpy as np

def input_data(trainDataple_dataset):
    return pd.read_csv(trainDataple_dataset,sep="\t",header=None)
  
def emission_matrix_calculate(trainData): 
    em = [[0,0],[0,0]]
    em[0][0]=trainData.count(['cheat','lose'])/(trainData.count(['cheat','lose'])+trainData.count(['cheat','win']))
    em[0][1]=trainData.count(['cheat','win'])/(trainData.count(['cheat','lose'])+trainData.count(['cheat','win']))
    em[1][0]=trainData.count(['fair','lose'])/(trainData.count(['fair','lose'])+trainData.count(['fair','win']))
    em[1][1]=trainData.count(['fair','win'])/(trainData.count(['fair','lose'])+trainData.count(['fair','win']))
    em = np.asarray(em) 
    return em

def transition_matrix_count(trainData):
    train=trainData[:,0:1]
    em = [[0,0],[0,0]]
    for i in range(len(train)-1):
        if train[i]=='cheat' and train[i+1]=='cheat' :
            em[0][0]=em[0][0]+1
        elif train[i]=='cheat' and train[i+1]=='fair' :
            em[0][1]=em[0][1]+1
        elif train[i]=='fair' and train[i+1]=='cheat' :
            em[1][0]=em[1][0]+1
        elif train[i]=='fair' and train[i+1]=='fair' :
            em[1][1]=em[1][1]+1
    
    a=(em[0][0]+em[0][1])
    b=(em[1][0]+em[1][1])
    em[0][0]=em[0][0]/a
    em[0][1]=em[0][1]/a
    em[1][0]=em[1][0]/b
    em[1][1]=em[1][1]/b
    em = np.asarray(em)
    return em
    
def input_observation(testingData):
    return pd.read_csv(testingData,header=None)
    
def viterbi(observations, emission_matrix, initial_pro, transition_matrix):
    cheat=[0.0]*len(observations)
    fair=[0.0]*len(observations)
    cheat=np.array(cheat)
    fair=np.array(fair)
    
    cheat[0]=np.log(initial_pro[0]*(emission_matrix[0][0] if observations[0] == 'lose' else emission_matrix[0][1]))
    fair[0]=np.log(initial_pro[1]*(emission_matrix[1][0] if observations[0] == 'lose' else emission_matrix[1][1]))
    for i in range(1,len(observations)):
        a = cheat[i - 1] + np.log(transition_matrix[0][0]) + np.log((emission_matrix[0][0] if observations[i] == 'lose' else emission_matrix[0][1]))
        b = fair[i - 1] + np.log(transition_matrix[1][0])+ np.log((emission_matrix[0][0] if observations[i] == 'lose' else emission_matrix[0][1]))
        cheat[i]=max(a,b)
        a = cheat[i - 1] + np.log(transition_matrix[0][1])+ np.log((emission_matrix[1][0] if observations[i] == 'lose' else emission_matrix[1][1]))
        b = fair[i - 1] + np.log(transition_matrix[1][1])+ np.log((emission_matrix[1][0] if observations[i] == 'lose' else emission_matrix[1][1]))
        fair[i]=max(a,b)

    final_output=[["cheat"]]*len(observations)
    final_output=np.array(final_output)
    fair_count=0
    cheat_count=0
    for i in range(len(observations)):
        if fair[i]>cheat[i]:
            final_output[i]=["fair"]
            fair_count=fair_count+1
    cheat_count=len(observations)-fair_count
    score=np.array([fair_count,cheat_count])
    return score,final_output
    
def main():
    sample_dataset = "training_Dragon_1000.data.txt"  
    testing_data = "testing_Dragon_1000.data.txt"     

    observations = input_observation(testing_data)
    observations=observations.to_numpy();
    initial_pro = np.array([0.5, 0.5])

    trainData = input_data(sample_dataset)
    
    trainData=trainData.to_numpy();
    transition_matrix = transition_matrix_count(trainData)
    trainData=trainData.tolist()
    emission_matrix = emission_matrix_calculate(trainData)
    
    print("Transition Matrix\n", transition_matrix)
    print("Emission Matrix\n", emission_matrix)

    score,final_output=viterbi(observations, emission_matrix, initial_pro, transition_matrix)
    #print(final_output)
    print("Fair= ",score[0])
    print("Cheat= ",score[1])

main()