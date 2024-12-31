import math

import dill
import src
import sys
from exastar.time_series import TimeSeries
from exastar.genome import dt_genome
from exastar.genome.component.dt_node import DTNode
from exastar.genome.visitor import graphviz_visitor_dt
import check_buy_sell
import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import tqdm

def save_genome(genome):
    graph = graphviz_visitor_dt.GraphVizVisitorDT("src", "cpt_best3", genome)
    test = graph.visit()
    test.render(directory=".")

"""
Temp file to run and display the best selected genomes, used for testing.
"""
def search_best(num):
    csv_filename = (["C:/Users/matts/Documents/RIT/exa-star-dt/src/short_data/val.csv"])
    input_series_names = ["Predicted_CPT", "CPT_VOl_CHANGE", "CPT_TURNOVER", "CPT_BA_SPREAD", "CPT_ILLIQUIDITY",
                          "CPT_MARKET_CAP", "Predicted_KMX", "KMX_VOl_CHANGE", "KMX_TURNOVER", "KMX_BA_SPREAD",
                          "KMX_ILLIQUIDITY", "KMX_MARKET_CAP", "Predicted_RHI", "RHI_VOl_CHANGE", "RHI_TURNOVER",
                          "RHI_BA_SPREAD", "RHI_ILLIQUIDITY", "RHI_MARKET_CAP", "Predicted_STLD", "STLD_VOl_CHANGE",
                          "STLD_TURNOVER", "STLD_BA_SPREAD", "STLD_ILLIQUIDITY", "STLD_MARKET_CAP", "Predicted_UHS",
                          "UHS_VOl_CHANGE", "UHS_TURNOVER", "UHS_BA_SPREAD", "UHS_ILLIQUIDITY", "UHS_MARKET_CAP"]
    # output_series_names = ["CPT", "STLD", "RHI", "KMX", "UHS"]
    output_series_names = ["CPT", "STLD", "RHI", "KMX", "UHS"]
    initial_series = TimeSeries.create_norm_from_csv(filenames=csv_filename, input_series=None,
                                                     normalize_series=input_series_names,
                                                     output_series=output_series_names)
    #
    csv_filename = (["C:/Users/matts/Documents/RIT/exa-star-dt/src/short_data/test.csv"])
    test_series = TimeSeries.create_norm_from_csv(filenames=csv_filename, input_series=None,
                                                     normalize_series=input_series_names,
                                                     output_series=output_series_names)
    # mypath = f'./output/test/combined{num}/'
    mypath = f'./output/test/short_20000_7/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    best = ["", math.inf]
    running_sum = []

    for f in tqdm.tqdm(onlyfiles):
        genome = f.split(".")
        if int(genome[0]) < 84000:
            with open(mypath + '/' + f, 'rb') as file:
                try:
                    loaded_data = dill.load(file)
                    fit = loaded_data.fitness.mse

                    if fit < .2:
                        profit = loaded_data.test_genome(initial_series, False)
                        # print(fit, profit, f)
                        # if profit < 0.7:
                        # profit = loaded_data.test_genome(test_series, False)
                        running_sum.append(profit.detach().numpy())
                        if profit < best[1]:
                            best[0] = f
                            best[1] = profit
                            print(f"BEST: {profit}, {fit}, {f}")
                except:
                    pass

    print(best)
    rs = np.array(running_sum)
    print(rs.mean())
    print(rs.std())
    print(loaded_data.test_genome(test_series, False))
    return(best)

def search_best_train(num):
    csv_filename = (["C:/Users/matts/Documents/RIT/exa-star-dt/src/short_data/val.csv"])
    input_series_names = ["Predicted_CPT", "CPT_VOl_CHANGE", "CPT_TURNOVER", "CPT_BA_SPREAD", "CPT_ILLIQUIDITY",
                          "CPT_MARKET_CAP", "Predicted_KMX", "KMX_VOl_CHANGE", "KMX_TURNOVER", "KMX_BA_SPREAD",
                          "KMX_ILLIQUIDITY", "KMX_MARKET_CAP", "Predicted_RHI", "RHI_VOl_CHANGE", "RHI_TURNOVER",
                          "RHI_BA_SPREAD", "RHI_ILLIQUIDITY", "RHI_MARKET_CAP", "Predicted_STLD", "STLD_VOl_CHANGE",
                          "STLD_TURNOVER", "STLD_BA_SPREAD", "STLD_ILLIQUIDITY", "STLD_MARKET_CAP", "Predicted_UHS",
                          "UHS_VOl_CHANGE", "UHS_TURNOVER", "UHS_BA_SPREAD", "UHS_ILLIQUIDITY", "UHS_MARKET_CAP"]
    # output_series_names = ["CPT", "STLD", "RHI", "KMX", "UHS"]
    output_series_names = ["CPT", "STLD", "RHI", "KMX", "UHS"]
    initial_series = TimeSeries.create_norm_from_csv(filenames=csv_filename, input_series=None,
                                                     normalize_series=input_series_names,
                                                     output_series=output_series_names)
    #
    csv_filename = (["C:/Users/matts/Documents/RIT/exa-star-dt/src/short_data/test.csv"])
    test_series = TimeSeries.create_norm_from_csv(filenames=csv_filename, input_series=None,
                                                     normalize_series=input_series_names,
                                                     output_series=output_series_names)
    # mypath = f'./output/test/combined{num}/'
    mypath = f'./output/test/short_20000_5/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    best = ["", math.inf]
    running_sum = []
        
    for f in tqdm.tqdm(onlyfiles):
        genome = f.split(".")
        if int(genome) > 84000:
            pass
        with open(mypath + '/' + f, 'rb') as file:
            try:
                loaded_data = dill.load(file)
                fit = loaded_data.fitness.mse
                if fit < best[1]:
                    best[0] = f
                    best[1] = fit
                    print(f"BEST: {fit}, {f}")
            except:
                pass
    print(best)
    rs = np.array(running_sum)
    print(rs.mean())
    print(rs.std())
    print(loaded_data.test_genome(test_series, False))
    return(best)
def main():
    # with open('./output/best_g/7819.genome', 'rb') as file:
    with open('./output/test/short_20000_9/60334.genome', 'rb') as file:
        loaded_data = dill.load(file)

        csv_filename = (["C:/Users/matts/Documents/RIT/exa-star-dt/src/short_data/test.csv"])
        input_series_names = ["Predicted_CPT", "CPT_VOl_CHANGE", "CPT_TURNOVER", "CPT_BA_SPREAD", "CPT_ILLIQUIDITY",
                                        "CPT_MARKET_CAP", "Predicted_KMX", "KMX_VOl_CHANGE", "KMX_TURNOVER", "KMX_BA_SPREAD",
                                        "KMX_ILLIQUIDITY", "KMX_MARKET_CAP", "Predicted_RHI", "RHI_VOl_CHANGE", "RHI_TURNOVER",
                                        "RHI_BA_SPREAD", "RHI_ILLIQUIDITY", "RHI_MARKET_CAP", "Predicted_STLD", "STLD_VOl_CHANGE",
                                        "STLD_TURNOVER", "STLD_BA_SPREAD", "STLD_ILLIQUIDITY", "STLD_MARKET_CAP", "Predicted_UHS",
                                        "UHS_VOl_CHANGE", "UHS_TURNOVER", "UHS_BA_SPREAD", "UHS_ILLIQUIDITY", "UHS_MARKET_CAP"]
        output_series_names = ["CPT", "STLD", "RHI", "KMX", "UHS"]
        # output_series_names = ["STLD"]
        initial_series = TimeSeries.create_norm_from_csv(filenames=csv_filename, input_series=None, normalize_series=input_series_names,
                                                              output_series=output_series_names)

        print(loaded_data.test_genome(initial_series, False))
        loss, history = loaded_data.test_genome_graph_org(initial_series, False)
        print(loss)
        loss_org, history_org = loaded_data.test_genome_graph(initial_series, False)
        print(loss_org)
        los, b_err, s_err = loaded_data.test_genome_daily(initial_series)
        print(los)
        print(b_err)
        print(s_err)
        print()
        save_genome(loaded_data)
        print("START STRAT")
        check_buy_sell.hold_stocks()
        print("Main")
        check_buy_sell.main()
        csv_dict = pd.read_csv("C:/Users/matts/Documents/RIT/exa-star-dt/src/short_data/test.csv", encoding="UTF-8")
        color = ["red", "blue", "green", "orange", "cyan", "purple"]
        count = 0
        fig = plt.figure()
        ax = plt.subplot(111)

        for f in history:
            temp = []
            for i in history[f]:
                if isinstance(i, int):
                    temp.append(i)
                else:
                    temp.append(i.item())
            print(f)
            if f != "Value":
                pass
                ax.plot(range(len(history[f])), temp, label=f + " Shares", color=color[count], linestyle='-',)
                count += 1
            # else:
            #     plt.plot(range(len(history[f])), temp, label=f, color=color[5], linestyle='-', )

        # temp = []
        # for i in history["STLD"]:
        #     temp.append(i.item())
        csv_dict = pd.read_csv("C:/Users/matts/Documents/RIT/exa-star-dt/src/short_data/test.csv", encoding="UTF-8")
        # plt.plot(range(len(history["STLD"])), temp, label='STLD', color=color[0], linestyle='-', )
        # ax.plot(range(len(history["STLD"])), csv_dict["CPT"][:-1], label='CPT Price', color=color[0], linestyle='-', )
        # ax.plot(range(len(history["STLD"])), csv_dict["STLD"][:-1], label='STLD Price', color=color[1], linestyle='-', )
        # ax.plot(range(len(history["STLD"])), csv_dict["RHI"][:-1], label='RHI Price', color=color[2], linestyle='-', )
        # ax.plot(range(len(history["STLD"])), csv_dict["KMX"][:-1], label='KMX Price', color=color[3], linestyle='-', )
        # ax.plot(range(len(history["STLD"])), csv_dict["UHS"][:-1], label='UHS Price', color=color[4], linestyle='-', )

        # plt.title("Run 10 of Decision Maker")
        # plt.xlabel("Timestep (Days)")
        # plt.ylabel("Shares Held")

        plt.title("Value of Stock Over Test Run")
        plt.xlabel("Timestep (Days)")
        plt.ylabel("Prices Held($)")

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  fancybox=True, shadow=True, ncol=5, fontsize=8)
        plt.show()
        # csv_dict = pd.read_csv("test.csv", encoding="UTF-8")
        # price = csv_dict["STLD"]
        # buy_sell = csv_dict["Predicted_STLD"]
        # run_sum = check_buy_sell.run(buy_sell, price)
        # print()
        # profit, b_err, s_err = loaded_data.test_genome_daily(initial_series)
        # print(float(profit))
        # print(b_err)
        # print(s_err)
        # print()
        # check_buy_sell.hold_stock("STLD")


        # print()
        # print("Compared to:")
        # check_buy_sell.main()


        print(loaded_data)  # Output: 25


def test():
    with open('./output/test/short_20000_9/60334.genome', 'rb') as file:
        loaded_data = dill.load(file)

        csv_filename = (["C:/Users/matts/Documents/RIT/exa-star-dt/src/short_data/test.csv"])
        input_series_names = ["Predicted_CPT", "Predicted_KMX", "Predicted_RHI", "Predicted_STLD", "Predicted_UHS"]
        output_series_names = ["CPT", "STLD", "RHI", "KMX", "UHS"]
        # output_series_names = ["STLD"]
        initial_series = TimeSeries.create_norm_from_csv(filenames=csv_filename, input_series=None, normalize_series=input_series_names,
                                                              output_series=output_series_names)
        # save_genome(loaded_data)

        loss, history = loaded_data.test_genome_graph_org(initial_series, False)

if __name__ == "__main__":
    test()
    # main()
    # search_best("")
    # search_best_train("")
    # for i in range(2,11):
    #     search_best(i)