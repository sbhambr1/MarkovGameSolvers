import os
import numpy as np
import matplotlib.pyplot as plt

def main():

    with open('random_uniform.npy', 'rb') as f:
        random_uniform = np.load(f)
    with open('random_non_uniform.npy', 'rb') as f:
        random_non_uniform = np.load(f)
    with open('expert_uniform.npy', 'rb') as f:
        expert_uniform = np.load(f)
    with open('expert_non_uniform.npy', 'rb') as f:
        expert_non_uniform = np.load(f)
    with open('updated_uniform.npy', 'rb') as f:
        updated_uniform = np.load(f)
    with open('updated_non_uniform.npy', 'rb') as f:
        updated_non_uniform = np.load(f)

    gamma = np.arange(0, 100, 5)/100.0
        
    random_non_uniform_results = []
    export_non_uniform_results = []
    updated_non_uniform_results = []
    for state in range(10):
        state_random_non_uniform_results = []
        state_expert_non_uniform_results = []
        state_updated_non_uniform_results = []
        for i in range(20):
            state_random_non_uniform_results.append(random_non_uniform[i][state])
            state_expert_non_uniform_results.append(expert_non_uniform[i][state])
            state_updated_non_uniform_results.append(updated_non_uniform[i][state])
        random_non_uniform_results.append(state_random_non_uniform_results)
        export_non_uniform_results.append(state_expert_non_uniform_results)
        updated_non_uniform_results.append(state_updated_non_uniform_results)

    root_dir = os.getcwd()
    dir_name = 'Non_Uniform_Cost_Model_Comparison'
    path = root_dir + '/src/zero_sum/plots/' + dir_name
    if not os.path.exists(path):
        os.makedirs(path)
    
    for i in range(10):
        plt.figure()
        plt.plot(gamma, random_non_uniform_results[i], label='MMP', linestyle='--', marker='o', color='cyan')
        plt.plot(gamma, export_non_uniform_results[i], label='URS', linestyle=':', marker='x', color='orange')
        plt.plot(gamma, updated_non_uniform_results[i], label='OPT', linestyle='-', marker='+', color='black')
        plt.legend(['Naive Model - randomly set', 'Naive Model - expert set', 'Updated Model with Tuned Parameters'])
        plt.title('State {}'.format(i))
        plt.xlabel('Discount Factor $\gamma \longrightarrow$')
        plt.ylabel("Defender's Utility $\mathcal{V}_D \longrightarrow$")
        plt.savefig(path + '/state_{}.png'.format(i))

if __name__ == '__main__':
    main()