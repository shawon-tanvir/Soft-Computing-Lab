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
    
def viterbi(V, a, b, initial_distribution):
    T = V.shape[0]
    M = len(a)
    print(M)
    omega = np.zeros((T, M))
    omega[0, :] = np.log(initial_distribution * b[:, V[0]])
 
    prev = np.zeros((T - 1, M))
 
    for t in range(1, T):
        for j in range(M):
            # Same as Forward Probability
            probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, V[t]])
 
            # This is our most probable state given previous state at time t (1)
            prev[t - 1, j] = np.argmax(probability)
 
            # This is the probability of the most probable state (2)
            omega[t, j] = np.max(probability)
 
    # Path Array
    S = np.zeros(T)
 
    # Find the most probable last hidden state
    last_state = np.argmax(omega[T - 1, :])
 
    S[0] = last_state
 
    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1
 
    # Flip the path array since we were backtracking
    S = np.flip(S, axis=0)
 
    # Convert numeric values to actual hidden states
    result = []
    for s in S:
        if s == 0:
            result.append("A")
        else:
            result.append("B")
 
    return result
 
    
    
def main():
    sample_dataset = "training_Dragon_1000.data.txt"  
    testing_data = "testing_Dragon_1000.data.txt"     

    observations = input_observation(testing_data)
    check=observations.to_numpy();
    
    observations=[0]*len(check)
    observations=np.array(observations)
    for i in range(len(check)):
        if check[i]=='lose':
            observations[i]=1
    print(observations)
    
    initial_pro = np.array([0.5, 0.5])

    trainData = input_data(sample_dataset)
    
    trainData=trainData.to_numpy();
    transition_matrix = transition_matrix_count(trainData)
    trainData=trainData.tolist()
    transition_matrix=transition_matrix.tolist()
    print(transition_matrix)
    emission_matrix = emission_matrix_calculate(trainData)
    emission_matrix=emission_matrix.tolist()
    print(emission_matrix)
    print("Transition Matrix\n", transition_matrix)
    print("Emission Matrix\n", emission_matrix)

    final_output=viterbi(observations, emission_matrix, initial_pro, transition_matrix)
    print(final_output)
    

main()