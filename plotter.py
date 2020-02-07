#!/usr/bin/python3

# File:     A program which parses the input data from the tests, and creates plots.
# Author:   Ardalan Ahanchi
# Date:     Winter 2020

import re                               #For regex pattern recognition.
import sys                              #To read from stdin.
import matplotlib.pyplot as plot        #For plotting.

#A class which represents an output entry which will be plotted.
class Entry:
    #Default constructor for setting all the values.
    def __init__(self, transfer_to_dev, transfer_to_host, calc_time_cuda, calc_time_seq, n, blocks, threads):
        self.transfer_to_dev = transfer_to_dev
        self.transfer_to_host = transfer_to_host
        self.calc_time_cuda = calc_time_cuda
        self.calc_time_seq = calc_time_seq
        self.n = n
        self.blocks = block
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
fig1_x_seq, fig1_x_cu = [], []
fig1_y_seq, fig1_y_cu = [], []

#Iterate through the data and filter out the right points for drawing.
for ent in data:
    if ent.? == ?:                           #Add the data points.
        fig1_x_seq.append(ent.?)
        fig1_y_seq.append(ent.?)


plot.plot(fig1_x_seq, fig1_y_seq, label="Sequential")
plot.plot(fig1_x_12t, fig1_y_12t, label="Cuda")
plot.legend()
plot.title("Vector Addition")
plot.xlabel("N")
plot.ylabel("Time (Seconds)")
plot.show()
