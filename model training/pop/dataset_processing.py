import os
import gc
import h5py
import pickle
import numpy as np

from functions import fDeltas
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset_type = np.dtype([('matrix_X', 'f8', (100, 13)), ('Y', 'b')])
def_parameter = "pop"
def_threshold = 0.1
# def_dataset_path = "dataset_mut_0.01.hdf5"
def_dataset_name = "test_dataset"


def whichFunction(file_name):
    file_name_list = file_name.split(sep="_")
    return int(file_name_list[2][3:])


def preprocess_dataset(data, test_size=0.15, random_state=1, pca_n=None):
    X = data["matrix_X"]
    y = data["Y"]

    X = X.reshape(X.shape[0], X[0].size)
    print(X.shape)
    
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    pickle.dump(scaler, open("std.pkl","wb"))
    
    # PCA begins
    if pca_n is not None:
        pca = PCA(n_components=pca_n)
        X = pca.fit_transform(X)
        pickle.dump(pca, open("pca.pkl","wb"))
    # PCA ends
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size, random_state=random_state)
    val_size = int(0.15 * X.shape[0])

    # 15% for validation
    # 15% for test
    # 70% for training

    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]
    return X_train, X_test, X_val, y_train, y_test, y_val


class DatasetDE:
    def __init__(self, parameter=def_parameter, threshold=def_threshold, dataset_name=def_dataset_name):
        self.parameter = parameter
        self.threshold = threshold
        self.dataset_name = dataset_name + "_" + parameter
        self.dataset_path = dataset_name + "_" + parameter + "_" + str(threshold) + ".hdf5"
        self.data = None

    def create_dataset(self):
        data_list = []
        y_count = [0, 0, 0]
        dir_path = "./experiment-data"
        print(f">>> Creating the dataset of threshold = {self.threshold} >>>")
        
        for exp_file in os.listdir(dir_path):
            if exp_file.startswith("."):
                continue
            # use pickle to get each stored Data_DE list
            root = pickle.load(open(os.path.join(dir_path, exp_file), "rb"))
            # for each Data_DE object in list "root"
            for item in root:
                # get which fitness function does it use
                fun_n = whichFunction(exp_file)
                if fun_n == 20:
                    continue
                # create a errors list that will be used to determine which index has the best performance(index of minimum)
                # res_list = [item.results[0], item.fun, item.results[1]]
                # result_dec = item.results[0]
                # result_inc = item.results[1]
                errors = [abs(item.fun - fDeltas[fun_n - 1]),  # original result   0
                          abs(item.results[0] - fDeltas[fun_n - 1]),  # decreased result  1
                          abs(item.results[1] - fDeltas[fun_n - 1])]  # increased result  2

                #  0: original  is the best
                #  1: decreased is the best
                #  2: increased is the best
                Y = errors.index(min(errors))

                if Y != 0 and errors[0] != 0:
                    bool1 = abs(errors[1] - errors[0]) / errors[0] < self.threshold
                    bool2 = abs(errors[2] - errors[0]) / errors[0] < self.threshold
                    if bool1 and bool2:
                        Y = 0
                y_count[Y] += 1
                # create a temporary tuple, which will be stored in a list
                tmp_tuple = (item.data, Y)
                data_list.append(tmp_tuple)
                # fun_count_list[fun_n] += 1
                # y_count_list[Y] += 1

        print(y_count)
        dataset_array = np.array(data_list, dtype=dataset_type)
        # Shuffle the dataset array to improve data distribution
        np.random.shuffle(dataset_array)
        print(">>> Dataset Creation Done >>>")
        return dataset_array

    def store_dataset(self, data):
        f = h5py.File(self.dataset_path, "w")
        f.create_dataset(self.dataset_name, dtype=dataset_type, data=data,
                         compression="gzip",compression_opts=9)
        print(">>> Dataset Storage Done >>>")
        f.close()
        return self.dataset_path


dataset_DE = DatasetDE("mut", 0.1)
dataset_DE.create_dataset()

dataset_DE = DatasetDE("mut", 0.15)
dataset_DE.create_dataset()

dataset_DE = DatasetDE("mut", 0.2)
dataset_DE.create_dataset()
