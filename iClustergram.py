"""
interactive clustergram!!

Author: Zichen Wang
Created on 4/8/2014
"""

import plotly
py = plotly.plotly("wangz10", "fmnoxd2t2u")

# import urllib2, StringIO, csv

# url = 'https://gist.github.com/chriddyp/8818473/raw/d8c73ff66a190a84eb8c6c19df4d8865673234ca/2007gapminder.csv'
# response = urllib2.urlopen(url).read()
# output = StringIO.StringIO(response)
# cr = csv.reader(output)
# def tryFloat(d):
#     try:
#         return float(d)
#     except ValueError:
#         return d

# data = [[tryFloat(dij) for dij in di] for di in cr]

# import math
# graph_data = [] # the data structure that will describe our plotly graph
# for continent in ['Asia', 'Europe', 'Africa', 'Americas', 'Oceania']:
#     '''
#         "x" data is GDP Per Capita
#         "y" data is Life Expectancy
#         "text" data is the Country Name
#         and we scale the "marker" size
#          to each country's population
#     '''
#     graph_data.append(
#     {
#         'name': continent, # the "name" of this series is the Continent
#         'x': [row[4] for row in data if row[2] == continent],
#         'y': [row[3] for row in data if row[2] == continent],
#         'text': [row[0] for row in data if row[2] == continent],

#         'type': 'scatter',
#         'mode': 'markers',

#         'marker': { # specify the style of the individual scatter points
#             'size': [math.sqrt(row[1])/1.e3 for row in data if row[2] == continent],
#             'sizemode': 'area',
#             'sizeref': 0.05,
#             'opacity': 0.55
#         }
#     })
# layout = {
#     'xaxis': {'title': 'GDP Per Capita'},
#     'yaxis': {'title': 'Life Expectancy'},
#     'title': 'Hans Rosling Bubble Chart<br>2007'
# }

# py.plot(graph_data, layout=layout,
#          filename='My first plotly graph', fileopt='overwrite',
#          world_readable=True, width=1000, height=650)
# {
#     "url": "https://plot.ly/~IPython.Demo/1085",
#     "filename": "My first plotly graph",
#     "error": "",
#     "warning": "",
#     "message": ""
# }


import random

zd=[random.sample(range(0, 400),400) for j in range(200)]


cs=[[0,"rgb(12,51,131)"],[0.25,"rgb(10,136,186)"],[0.5,"rgb(242,211,56)"],
	[0.75,"rgb(242,143,56)"],[1,"rgb(217,30,30)"]]
		
py.plot([{'z': zd,'scl':cs,'type': 'heatmap'}])


