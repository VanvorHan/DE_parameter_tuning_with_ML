import os
import sys
import pickle
import numpy as np

def whichFunction(file_name):
    file_name_list = file_name.split(sep="_")
    return int(file_name_list[2][3:])


data_path = "./experiment-data"
fun_count_arr = np.zeros(shape=(28,), dtype=int)

print("DE-pop Summary: ")
for exp_file in os.listdir(data_path):
    if exp_file.startswith("."):
        continue
    exp_file_path = os.path.join(data_path, exp_file)
    fun_num = whichFunction(exp_file)
    root = pickle.load(open(exp_file_path, "rb"))
    fun_count_arr[fun_num - 1] += len(root)

it = np.nditer(fun_count_arr, flags=['f_index'])
for fun_count in it:
    print(f"Function {it.index + 1}: {fun_count}")

print("Count Array: ", fun_count_arr)
print(f"sum: {sum(fun_count_arr)}")
