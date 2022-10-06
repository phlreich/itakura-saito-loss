import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_reliability(freqs, save_path="reliability_plot.svg", save=True):

    """
    expects torch.tensor of size 10 with missing values set to nan
    """
    nan = float("nan")
    df = pd.DataFrame(freqs.cpu()).transpose()
    df.columns = df.columns / 10

    plt.clf()

    main_thickness = 1.8

    # set square aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot([0,1], [0,1], color="grey", linestyle="--", linewidth=3.05, label='_nolegend_')
    # set background grey grid with dotted lines with offset
    plt.grid(color='grey', linestyle='dotted', linewidth=1, alpha=1)

    # plot the actual output
    plt.bar(df.columns, 
        df.iloc[0], 
        width=0.1, 
        align="edge", 
        edgecolor="black", 
        color="blue",
        alpha=1, 
        zorder=2)

    # plot the expected output
    plt.bar(df.columns,
        [n if not pd.isna(df.iloc[0].iloc[i]) else 0 for i, n in enumerate(np.arange(0.05, 1, 0.1))],
        width=0.1, 
        align="edge",
        edgecolor=matplotlib.colors.colorConverter.to_rgba('red', alpha=1),
        color=matplotlib.colors.colorConverter.to_rgba('red', alpha=0.36), 
        zorder=1)


    plt.ylabel("Accuracy")
    plt.xlabel("Confidence")
    plt.ylim(0,1)
    plt.xlim(0,1)

    # set gridlines below bars (except alpha < 1)
    plt.gca().set_axisbelow(True)
    plt.rcParams.update({'font.size': 14})

    # change edge thickness
    for i, bar in enumerate(plt.gca().patches):
        bar.set_linewidth(main_thickness)
        if i <= 10:
            bar.set_linewidth(main_thickness)

    # set all axis color to grey
    plt.gca().spines["bottom"].set_color('grey')
    plt.gca().spines["top"].set_color('grey')
    plt.gca().spines["right"].set_color('grey')
    plt.gca().spines["left"].set_color('grey')

    # set axis thickness to main_thickness
    main_thickness += 0.3
    plt.gca().spines["bottom"].set_linewidth(main_thickness)
    plt.gca().spines["top"].set_linewidth(main_thickness)
    plt.gca().spines["right"].set_linewidth(main_thickness)
    plt.gca().spines["left"].set_linewidth(main_thickness)

    # set background grid thickness
    plt.grid(linewidth=1.7)


    plt.tick_params("both", length=0, pad=8)


    # add boxed legend to top left inside graph with backgroud color white
    plt.legend(["Outputs", "Gap", ], loc="upper left", bbox_to_anchor=(0.01, 0.99), frameon=True, facecolor="white", framealpha=1,)

    # adjust bottom to avoid cutoff
    plt.subplots_adjust(bottom=0.15)

    if save:
        plt.savefig(save_path, format="svg")

