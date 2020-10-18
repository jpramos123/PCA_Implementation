from utils import *
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


if __name__ == "__main__":

    
    mat2 = [[2.5,  2.4],
            [0.5, 0.7],
            [2.2,2.9],
            [1.9, 2.2],
            [3.1, 3.0],
            [2.3, 2.7],
            [2, 1.6],
            [1, 1.1],
            [1.5, 1.6],
            [1.1, 0.9]
            ]

        

    print("** Class Example **")

    data = pd.DataFrame(mat2)
    data = data.values.tolist()

    data_msm = pd.DataFrame(mat2)
    data_msm = data_msm.values.tolist()

    data_msm = mean_data_adjust(transpose(data_msm))
    data_msm = transpose(data_msm)
    data_msm = buildData(pd.DataFrame(data_msm))
    
    l = linear(data_msm)

    ein_arr = makePCA(data)[0][1]
    ein_val = makePCA(data)[1][1]
    print("Eigenvector: ", ein_arr)
    print("Eingenvalue: ", ein_val)


    adj_data = mean_data_adjust(transpose(data))
    adj_data = transpose(adj_data)
    plt.title('Class Example')
    plt.scatter([adj_data[i][0] for i in range(len(adj_data))],[adj_data[i][1] for i in range(len(adj_data))], c='r')
    plt.plot(l[0],l[1], label='MSM')
    plt.plot([-1.5,ein_arr[0]*5],[-1.5,ein_arr[1]*5], label='PCA')
    plt.xlim(-1.6,2.6)
    plt.ylim(-1.6,2.6)
    plt.legend()


    print("** US-Census **")

    data = pd.read_csv("US-Census.txt", delim_whitespace=True)
    data = data.values.tolist()
    
    data_msm = pd.read_csv("US-Census.txt", delim_whitespace=True)
    data_msm = data_msm.values.tolist()

    data_msm = mean_data_adjust(transpose(data_msm))
    data_msm = transpose(data_msm)
    data_msm = buildData(pd.DataFrame(data_msm))
    
    l = linear(data_msm)

    ein_arr = makePCA(data)[0][1]
    ein_val = makePCA(data)[1][1]
    print("Eigenvector: ", ein_arr)
    print("Eingenvalue: ", ein_val)

    adj_data = mean_data_adjust(transpose(data))
    adj_data = transpose(adj_data)


    plt.figure()
    plt.title('US-Census')
    plt.scatter([adj_data[i][0] for i in range(len(adj_data))],[adj_data[i][1] for i in range(len(adj_data))], c='r')
    plt.plot(l[0],l[1], label='MSM')
    plt.plot([-100,ein_arr[0]*150],[-200,ein_arr[1]*150], label='PCA')
    plt.xlim(-60,65)
    plt.ylim(-110,135)
    plt.legend()


    print("** Alpswater **")

    data = pd.read_csv("alpswater.txt", delim_whitespace=True)
    data.drop(columns=['Row'], inplace=True)
    data = data.values.tolist()

    data_msm = pd.read_csv("alpswater.txt", delim_whitespace=True)
    data_msm.drop(columns=['Row'], inplace=True)
    data_msm = data_msm.values.tolist()

    data_msm = mean_data_adjust(transpose(data_msm))
    data_msm = transpose(data_msm)
    data_msm = buildData(pd.DataFrame(data_msm))
    
    l = linear(data_msm)

    ein_arr = makePCA(data)[0][1]
    ein_val = makePCA(data)[1][1]
    print("Eigenvector: ", ein_arr)
    print("Eingenvalue: ", ein_val)

    adj_data = mean_data_adjust(transpose(data))
    adj_data = transpose(adj_data)


    plt.figure()
    plt.title('Alpswater')
    plt.scatter([adj_data[i][0] for i in range(len(adj_data))],[adj_data[i][1] for i in range(len(adj_data))], c='r')
    plt.plot([-50,ein_arr[0]*50],[-100,ein_arr[1]*50], label='PCA')
    plt.plot(l[0],l[1], label='MSM')
    plt.xlim(-8,10)
    plt.ylim(-18,18)
    plt.legend()


    print("** Books Attend Grade **")

    data = pd.read_csv("Books_attend_grade.txt", delim_whitespace=True)

    data = data.values.tolist()
    
    ein_arr = makePCA(data)[0][1]
    ein_val = makePCA(data)[1][1]
    print("Eigenvector: ", ein_arr)
    print("Eingenvalue: ", ein_val)

    plt.show()