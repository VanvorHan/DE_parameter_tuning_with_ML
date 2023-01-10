import os
import sys
import pickle
import functions
import numpy as np
import tensorflow as tf

from decimal import *
from data_DE import DataDE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from differential_evolution import differential_evolution

fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]
## 28 functions

dim = 30
max_evals = 10_000 * dim  # like in the CEC'13 benchmark
cec_benchmark = functions.CEC_functions(dim)

bound = 100
bounds = dim * [(bound, -bound)]
strategy = 'rand1bin'
pre_change = Decimal('0.01')
pop_pre_change = 1

def obj_function(X):
    return cec_benchmark.Y(X, fun_num)

# Where the program starts
run_num = 50
result_list = []
thresholds = [0.1, 0.15, 0.2]
test_functions = [2, 7, 12, 19, 24]
param_names = ["mut", "rec", "pop"]

for threshold in thresholds:
    fun_list = []
    models = []
    for param in param_names:
        model_name = param + "_models/" + param + "_NN_t" + str(threshold)
        model = tf.keras.models.load_model(model_name)
        models.append(model)
    #models = [mut_model, rec_model, pop_model]
    
    for fun_num in test_functions:
        s = "**** function " + str(fun_num) + ": ****"
        print(s)
        errors = []
        #recombination = 0.9
        #popsize = 30
        #mutation = 0.8
        parameters = [0.8, 0.9, 30]
        predictions = [-1,-1,-1]
        # 0: mut; 1: rec; 2:pop
        
        for run in range(1,run_num + 1):
            print(f"\t# {run} Run: ")
            for index, last_predict in enumerate(predictions):
                if index == 2:
                    if last_predict == 1:
                        parameters[index] -= pop_pre_change
                    elif last_predict == 2:
                        parameters[index] += pop_pre_change
                else:
                    getcontext().prec = 2
                    param_dec = Decimal(str(parameters[index]))
                    if last_predict == 1:
                        param_dec -= pre_change
                    elif last_predict == 2:
                        param_dec += pre_change
                    parameters[index] = float(param_dec)

            mutation = parameters[0]
            recombination = parameters[1]
            popsize = parameters[2]

            max_iter = int(max_evals / (popsize * dim))
            result = differential_evolution(obj_function, bounds=bounds, popsize=popsize,
                                    recombination=recombination, mutation=mutation,
                                    maxiter=max_iter, strategy=strategy, polish=False, tol=0)
            error = result.fun - fDeltas[fun_num - 1]
            getcontext().prec = 100
            error = float(Decimal(str(error)).quantize(Decimal('1.0000')))
            record = (parameters, error)
            print(f"\t\tparameters[mut, rec, pop]: {parameters}\n\t\terror:\t\t\t    {error}")
            errors.append(record)
        
            data = DataDE(result, mutation, recombination, popsize)
            X = data.data
            X = X.reshape(1,-1)
            
            for model in models:
                tmp_X = X
                index = models.index(model)
                param_name = param_names[index]
                std_pkl_path = os.path.join("scalers", param_name, "std.pkl")
                std_reload = pickle.load(open(std_pkl_path,'rb'))
                tmp_X = std_reload.transform(tmp_X)
            
                pca_pkl_path = os.path.join("scalers", param_name, "pca_" + str(model.input_shape[1]) + ".pkl")
                pca_reload = pickle.load(open(pca_pkl_path,'rb'))
                tmp_X = pca_reload.transform(tmp_X)
                
                pred = model.predict(np.array(tmp_X))
                last_predict = np.argmax(pred, axis=1)
                predictions[index] = last_predict[0]
            print(f"\t\tpredictions[mut, rec, pop]: {predictions}")

        fun_list.append(errors)

    errors_array = np.array(fun_list)
    print(errors_array)
    with open(f'together_results_t{threshold}.pickle', 'wb') as f:
        pickle.dump(errors_array, f)

