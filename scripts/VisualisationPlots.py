
import sys
sys.path.insert(0, '/AB_Testing')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class VisualiseDf:
    def __init__(self, df):
        self.df = df.copy()

    # plots a histogram
    # parameters: dataframe, column title, color (of hist)
    # returns: histogram plot (in the color green by default)
<<<<<<< HEAD
    def plot_hist(self, df:pd.DataFrame, column:str, color:str='cornflowerblue')->None:
        sns.displot(data=df, x=column, color=color, kde=True, height=5, aspect=2)
        plt.xticks(rotation=75, fontsize=14)
        plt.yticks( fontsize=14)
=======
    def plot_hist(df:pd.DataFrame, column:str, color:str)->None:
        sns.displot(data=df, x=column, color=color, kde=True, height=5, aspect=2)
>>>>>>> 9f2ddda94590d9255fbd63b0b6008634772ff986
        plt.title(f'Distribution of {column}', size=20, fontweight='bold')
        plt.show()

    # # plots the distribution from a histogram plot, where the data is ordered by hour
    # def plot_dist(self, df:pd.DataFrame,)->None:
    #     hours_bin_values = np.arrange(start=0, stop=23, step=1)
    #     np.histogram
    

    # plots a bar graph
    # parameters: dataframe, dependent col, independent col, xlabel, ylabel
    # returns: plot of bar graph
    def plot_bar(self, df:pd.DataFrame, x_col:str, y_col:str, title:str, xlabel:str, ylabel:str)->None:
        plt.figure(figsize=(12, 7))
        sns.barplot(data = df, x=x_col, y=y_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.yticks( fontsize=14)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.show()

    def plot_heatmap(df:pd.DataFrame, title:str, cbar=False)->None:
        plt.figure(figsize=(12, 7))
        sns.heatmap(df, annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.2f', linewidths=.7, cbar=cbar )
        plt.title(title, size=18, fontweight='bold')
        plt.show()

    # plots a box plot
    # parameters: dataframe, dependent col, title of box plot
    # returns: a box plot
    def plot_box(self, df:pd.DataFrame, x_col:str, title:str) -> None:
        plt.figure(figsize=(12, 7))
        sns.boxplot(data = df, x=x_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.show()

    
    def plot_box_multi(df:pd.DataFrame, x_col:str, y_col:str, title:str) -> None:
        plt.figure(figsize=(12, 7))
        sns.boxplot(data = df, x=x_col, y=y_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.yticks( fontsize=14)
        plt.show()

    def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, hue: str, style: str) -> None:
        plt.figure(figsize=(12, 7))
        sns.scatterplot(data = df, x=x_col, y=y_col, hue=hue, style=style)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.yticks( fontsize=14)
        plt.show()
        
    def plot_count(df: pd.DataFrame, column: str) -> None:
        plt.figure(figsize=(12, 7))
        sns.countplot(data=df, x=column)
        plt.title(f'Distribution of {column}', size=20, fontweight='bold')
        plt.show()

