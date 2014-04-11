"""
interactive clustergram!!!

Author: Zichen Wang
Created on 4/8/2014
"""

import plotly
import random, string

py = plotly.plotly("wangz10", "fmnoxd2t2u")

zd = [random.sample(range(0, 26),26) for j in range(26)]
xlabels = list(string.ascii_lowercase)
ylabels = list(string.ascii_uppercase)

cs=[[0,"rgb(12,51,131)"],[0.25,"rgb(10,136,186)"],[0.5,"rgb(242,211,56)"],
	[0.75,"rgb(242,143,56)"],[1,"rgb(217,30,30)"]]

d = {'x': xlabels, 'y': ylabels,'z': zd,'scl':cs,'type': 'heatmap'}

py.plot([d])


