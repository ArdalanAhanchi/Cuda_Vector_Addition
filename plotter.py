#!/usr/bin/python3

# File:     A program which parses the input data from the tests, and creates plots.
# Author:   Ardalan Ahanchi
# Date:     Winter 2020

import re                               #For regex pattern recognition.
import sys                              #To read from stdin.
import matplotlib.pyplot as plot        #For plotting.
from mpl_toolkits import mplot3d        #For 3D stuff.

#A class which represents an output entry which will be plotted.
class Entry:
    #Default constructor for setting all the values.
    def __init__(self, transfer_to_dev, transfer_to_host, calc_time_cuda, calc_time_seq, n, blocks, threads):
        self.transfer_to_dev = transfer_to_dev
        self.transfer_to_host = transfer_to_host
        self.calc_time_cuda = calc_time_cuda
        self.calc_time_seq = calc_time_seq
        self.n = n
        self.blocks = blocks
        self.threads = threads

#Create a regex pattern for matching the data from the output.
regex_pattern = '\[Cuda_Transfer_To_Device_Seconds\]=(\d.\d+e[-+]\d+).+\[Cuda_Transfer_To_Host_Seconds\]=(\d.\d+e[-+]\d+).+\[Cuda_Calculation_Time_Seconds\]=(\d.\d+e[-+]\d+).+\[Sequential_Time_Seconds\]=(\d.\d+e[-+]\d+).+\[N\]=(\d+).+\[Blocks\]=(\d+).+\[Threads\]=(\d+)'

#Create an array for holding the parsed data entries.
data = []

#Iterate through every line in stdin to read data.
for line in sys.stdin:
    #Match the pattern to the current line.
    regex_matched = re.match(regex_pattern, line)

    #If it matched, store the data in lists.
    if regex_matched:
        curr_entry = Entry(float(regex_matched.group(1)), float(regex_matched.group(2)),\
                            float(regex_matched.group(3)), float(regex_matched.group(4)),\
                            int(regex_matched.group(5)), int(regex_matched.group(6)),\
                            int(regex_matched.group(7)))

        data.append(curr_entry)
    else:
        print("Error: Could not parse data.")


#Figure 1
##########################################################################################
fig1_x_seq, fig1_x_cu_total, fig1_x_cu_calc_only = [], [], []
fig1_y_seq, fig1_y_cu_total, fig1_y_cu_calc_only = [], [], []
fig1_z_seq, fig1_z_cu_total, fig1_z_cu_calc_only = [], [], []

#Iterate through the data and filter out the right points for drawing.
for ent in data:
    #    fig1_x_seq.append(ent.calc_time_seq)
    #    fig1_y_seq.append(ent.blocks)
    #    fig1_z_seq.append(ent.threads)

    fig1_x_cu_total.append(ent.calc_time_cuda + ent.transfer_to_dev + ent.transfer_to_host)
    fig1_y_cu_total.append((ent.blocks * ent.threads))
    fig1_z_cu_total.append(ent.n)

    fig1_x_cu_calc_only.append(ent.calc_time_cuda)
    fig1_y_cu_calc_only.append((ent.blocks * ent.threads))
    fig1_z_cu_calc_only.append(ent.n)


#Create the 3d plot.
fig = plot.figure()
ax = plot.axes(projection='3d')

#Set labels and plot it.
plot.title("CUDA Total Time")
ax.scatter3D(fig1_z_cu_total, fig1_y_cu_total, fig1_x_cu_total)
ax.set_xlabel("N (Size)")
ax.set_ylabel("Total Threads (Threads * Blocks)")
ax.set_zlabel("Time (Seconds)")
plot.show()

plot.title("CUDA Calculations Only Time")
ax.scatter3D(fig1_z_cu_calc_only, fig1_y_cu_calc_only, fig1_x_cu_calc_only)
ax.set_xlabel("N (Size)")
ax.set_ylabel("Total Threads (Threads * Blocks)")
ax.set_zlabel("Time (Seconds)")
plot.show()
