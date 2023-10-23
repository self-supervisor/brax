import datetime
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

now = datetime.datetime.now()
files = glob.glob("/home/augustine/sac_analysis/????-??-??-??-??-??.csv")

recent_file_list = []
for file in files:
    filtered_file = file.split("/")[-1]
    year, month, day, hour, minute, second = map(
        int,
        filtered_file.split("-")[:-1] + [filtered_file.split("-")[-1].split(".")[0]],
    )
    file_datetime = datetime.datetime(year, month, day, hour, minute, second)

    if (now - file_datetime).total_seconds() <= 12 * 60 * 60:
        use_lff = pd.read_csv(file)["use_lff"]
        if use_lff[0] == True:
            recent_file_list.append(file)

lff_scale_list = [
    0.000001,
    0.0000025,
    0.000005,
    0.0000075,
    0.00001,
    0.000025,
    0.00005,
    0.000075,
    0.0001,
    0.00025,
    0.0005,
    0.00075,
    0.001,
    0.0025,
    0.005,
    0.0075,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    25.0,
    50.0,
    75.0,
    100.0,
]

mean_list = []
std_error_list = []
found_scale_list = []
for scale in lff_scale_list:
    scale_results = []
    for recent_file in recent_file_list:
        data = pd.read_csv(recent_file)
        if data["lff_scale"][0] == scale:
            scale_results.append(data["eval/episode_reward"][2])
    print(scale)
    print(scale_results)
    if len(scale_results) == 0:
        continue
    else:
        found_scale_list.append(scale)
    mean = sum(scale_results) / len(scale_results)
    std_err = np.std(scale_results) / np.sqrt(len(scale_results))
    mean_list.append(mean)
    std_error_list.append(std_err)

plt.errorbar(
    np.log(found_scale_list), mean_list, yerr=std_error_list, fmt="o", capsize=5
)
plt.xlabel("log LFF scale")
plt.ylabel("Average reward")
plt.title("Walker2d, LFF scale vs. average reward")
plt.savefig("walker2d_LFF_scale_vs_average_reward.png")
plt.show()
plt.close()
