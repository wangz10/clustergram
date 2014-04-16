"""
Interactive clustergram ploted with plotly API, https://plot.ly/
Users need to supply the function with username and APIkey for plotly
to enable this feature. 

TODOs:
Group labels are not supported yet.
Dendrogram can not be displayed, 
Colormaps haven't been costomized...

Author: Zichen Wang
Created on 4/8/2014
"""

import plotly
import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import zscore

def iClustergram(data=None, row_labels=None, col_labels=None,
			row_groups=None, col_groups=None,
			row_linkage='average', col_linkage='average', 
			row_pdist='euclidean', col_pdist='euclidean',
			standardize=None, log=False, 
			display_range=3, username='wangz10', apikey='fmnoxd2t2u'):
	## preprocess data
	if log:
		data = np.log2(data + 1.0)

	if standardize == 1: # Standardize along the columns of data
		data = zscore(data, axis=0)
	elif standardize == 2: # Standardize along the rows of data
		data = zscore(data, axis=1)

	## cluster data:
	## compute pdist for rows
	d1 = dist.pdist(data, metric=row_pdist)
	D1 = dist.squareform(d1)
	Y1 = sch.linkage(D1, method=row_linkage, metric=row_pdist)
	Z1 = sch.dendrogram(Y1, orientation='right')
	idx1 = Z1['leaves']

	## compute pdist for cols
	d2 = dist.pdist(data.T, metric=col_pdist)
	D2 = dist.squareform(d2)
	Y2 = sch.linkage(D2, method=col_linkage, metric=col_pdist)
	Z2 = sch.dendrogram(Y2)
	idx2 = Z2['leaves']

	## transform the orders of data to clustered data
	data_clustered = data
	data_clustered = data_clustered[:,idx2]
	data_clustered = data_clustered[idx1,:]
	data_to_plot = data_clustered.tolist()

	## transform the orders of row and col labels
	new_row_labels = []
	new_col_labels = []
	for i in range(data.shape[0]):
		new_row_labels.append(row_labels[idx1[i]])
	for i in range(data.shape[1]):
		new_col_labels.append(col_labels[idx2[i]])
	## plot clustered data using plotly
	py = plotly.plotly(username, apikey)
	d = {}
	d['x'] = new_row_labels
	d['y'] = new_col_labels
	d['z'] = data_to_plot
	d['type'] = 'heatmap'
	py.plot([d])
	return
