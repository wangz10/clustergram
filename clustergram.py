"""
A clustergram function similar to MATLAB clustergram()

Author: Zichen Wang
Created on 4/7/2014

References:

https://code.activestate.com/recipes/578834-hierarchical-clustering-heatmap-python/
http://www.mathworks.com/help/bioinfo/ref/clustergram.html

"""

import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import zscore

def clustergram(data=None, row_labels=None, col_labels=None,
			row_groups=None, col_groups=None,
			row_linkage='average', col_linkage='average', 
			row_pdist='euclidean', col_pdist='euclidean',
			standardize=None, log=False, colormap='redbluecmap',
			display_range=3):
	## preprocess data
	if log:
		data = np.log2(data + 1.0)

	if standardize == 1: # Standardize along the columns of data
		data = zscore(data, axis=0)
	elif standardize == 2: # Standardize along the rows of data
		data = zscore(data, axis=1)

	if colormap == 'redbluecmap':
		cmap = plt.cm.bwr
	elif colormap == 'redgreencmap':
		cmap = plt.cm.RdYlGn

	### Configure the Matplotlib figure size
	default_window_hight = 12
	default_window_width = 12
	fig = plt.figure(figsize=(default_window_width, default_window_hight)) ### could use m,n to scale here
	color_bar_w = 0.01 
	group_bar_w = 0.01
	heatmap_w = 0.6
	heatmap_h = 0.6
	dendrogram_l = 0.15
	color_legend_w = 0.18
	color_legend_h = 0.09
	margin = 0.01
	fig_margin = 0.10
	## calculate positions for all elements
	# ax1, placement of dendrogram 1, on the left of the heatmap
	rect1 = [fig_margin, fig_margin, dendrogram_l, heatmap_h]

	# axr, placement of row side colorbar
	rectr = [fig_margin + dendrogram_l, fig_margin, color_bar_w, heatmap_h]

	# axc, placement of column side colorbar
	rectc = [fig_margin + dendrogram_l + group_bar_w + margin, heatmap_h + fig_margin + margin, heatmap_w, color_bar_w]

	# axm, placement of heatmap
	rectm = [fig_margin + dendrogram_l + group_bar_w + margin, fig_margin, heatmap_w, heatmap_h]

	# ax2, placement of dendrogram 2, on the top of the heatmap
	rect2 = [fig_margin + dendrogram_l + group_bar_w + margin, fig_margin + heatmap_w + group_bar_w, heatmap_w, dendrogram_l] ### last one controls hight of the dendrogram

	# axcb - placement of the color legend
	rectcb = [0.05,0.90,0.18,0.09]

	## compute pdist for rows:
	d1 = dist.pdist(data, metric=row_pdist)
	D1 = dist.squareform(d1)
	ax1 = fig.add_axes(rect1, frame_on=True)
	Y1 = sch.linkage(D1, method=row_linkage, metric=row_pdist)
	Z1 = sch.dendrogram(Y1, orientation='right')
	idx1 = Z1['leaves']
	ax1.set_xticks([])
	ax1.set_yticks([])

	## compute pdist for cols
	d2 = dist.pdist(data.T, metric=col_pdist)
	D2 = dist.squareform(d2)
	ax2 = fig.add_axes(rect2, frame_on=True)
	Y2 = sch.linkage(D2, method=col_linkage, metric=col_pdist)
	Z2 = sch.dendrogram(Y2)
	idx2 = Z2['leaves']
	ax2.set_xticks([])
	ax2.set_yticks([])

	## plot color legend
	display_range = float(display_range)
	### Scale the max and min colors so that 0 is white/black
	vmin = data.min()
	vmax = data.max()
	vmax = max([vmax,abs(vmin)])
	vmin = vmax*-1
	norm = mpl.colors.Normalize(-display_range, display_range)
	axcb = fig.add_axes(rectcb, frame_on=False)
	step = display_range/3
	bounds = np.arange(-display_range, display_range+step, step)
	cb = mpl.colorbar.ColorbarBase(axcb, cmap=cmap, norm=norm, 
		orientation='horizontal', ticks=bounds, spacing='proportional', extend='both')
	axcb.set_title("colorkey")

	## plot heatmap
	axm = fig.add_axes(rectm)
	data_clustered = data
	data_clustered = data_clustered[:,idx2]
	data_clustered = data_clustered[idx1,:]

	im = axm.matshow(data_clustered, aspect='auto', origin='lower',cmap=cmap, norm=norm)
	axm.set_xticks([])
	axm.set_yticks([])

	## add labels
	new_row_header=[]
	new_column_header=[]
	for i in range(data.shape[0]):
		if len(row_labels) < 100:
			axm.text(data.shape[1]-0.5, i, '  '+row_labels[idx1[i]])
		new_row_header.append(row_labels[idx1[i]])
	for i in range(data.shape[1]):
		if len(col_labels) < 100:
			axm.text(i, -0.8, ' '+ col_labels[idx2[i]], rotation=270, verticalalignment="top") # rotation could also be degrees
		new_column_header.append(col_labels[idx2[i]])

	## plot group colors
	# numerize group names
	uniq_row_groups = list(set(row_groups))
	d_row_group = {} 
	for i, group_name in enumerate(uniq_row_groups):
		d_row_group[group_name] = i

	uniq_col_groups = list(set(col_groups))
	d_col_group = {} 
	for i, group_name in enumerate(uniq_col_groups):
		d_col_group[group_name] = i
	
	# cmap_group = mpl.colors.ListedColormap(['r', 'g', 'b', 'y', 'w', 'k', 'm'])
	cmap_col_groups = plt.cm.flag
	cmap_row_groups = plt.cm.prism

	axr = fig.add_axes(rectr)
	new_row_group = np.array([d_row_group[row_groups[idx1[i]]] for i in range(data.shape[0])])
	new_row_group.shape = (len(idx1), 1)
	axr.matshow(new_row_group, aspect='auto', origin='lower', cmap=cmap_row_groups)
	axr.set_xticks([])
	axr.set_yticks([])

	axc = fig.add_axes(rectc)
	new_col_group = np.array([d_col_group[col_groups[idx2[i]]] for i in range(data.shape[1])])
	new_col_group.shape = (1, len(idx2))	
	axc.matshow(new_col_group, aspect='auto', origin='lower', cmap=cmap_col_groups)
	axc.set_xticks([])
	axc.set_yticks([])

	## axgl for group label legends
	axgl = fig.add_axes([0.8,0.5,0.15,0.3])
	circle1 = mpl.patches.Circle((0,0), radius=0.1, color='#EB70AA')
	axgl.add_patch(circle1)
	axgl.set_xticks([])
	axgl.set_yticks([])


	plt.show()


## test:

data = np.arange(0,14,1).reshape(2,7)

clustergram(data=np.random.rand(4,3), row_labels=['a','c','e','d'], col_labels=['1','2','3'],
			row_groups=['A','B','C','C'], col_groups=['1','1','2'],
			row_linkage='average', col_linkage='average', 
			row_pdist='euclidean', col_pdist='euclidean',
			standardize=3, log=False, colormap='redbluecmap',
			display_range=3)
