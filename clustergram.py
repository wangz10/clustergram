"""
A clustergram function similar to MATLAB clustergram()

Author: Zichen Wang
Created on 4/7/2014

References:

https://code.activestate.com/recipes/578834-hierarchical-clustering-heatmap-python/
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


	### Scale the Matplotlib window size
	default_window_hight = 8.5
	default_window_width = 12
	fig = plt.figure(figsize=(default_window_width, default_window_hight)) ### could use m,n to scale here
	color_bar_w = 0.015 ### Sufficient size to show

	## calculate positions for all elements
	# ax1, placement of dendrogram 1, on the left of the heatmap
	#if row_method != None: w1 = 
	[ax1_x, ax1_y, ax1_w, ax1_h] = [0.05,0.22,0.2,0.6]   ### The second value controls the position of the matrix relative to the bottom of the view
	width_between_ax1_axr = 0.004
	height_between_ax1_axc = 0.004 ### distance between the top color bar axis and the matrix

	# axr, placement of row side colorbar
	[axr_x, axr_y, axr_w, axr_h] = [0.31,0.1,color_bar_w,0.6] ### second to last controls the width of the side color bar - 0.015 when showing
	axr_x = ax1_x + ax1_w + width_between_ax1_axr
	axr_y = ax1_y; axr_h = ax1_h
	width_between_axr_axm = 0.004

	# axc, placement of column side colorbar
	[axc_x, axc_y, axc_w, axc_h] = [0.4,0.63,0.5,color_bar_w] ### last one controls the hight of the top color bar - 0.015 when showing
	axc_x = axr_x + axr_w + width_between_axr_axm
	axc_y = ax1_y + ax1_h + height_between_ax1_axc
	height_between_axc_ax2 = 0.004

	# axm, placement of heatmap for the data matrix
	[axm_x, axm_y, axm_w, axm_h] = [0.4,0.9,2.5,0.5]
	axm_x = axr_x + axr_w + width_between_axr_axm
	axm_y = ax1_y; axm_h = ax1_h
	axm_w = axc_w

	# ax2, placement of dendrogram 2, on the top of the heatmap
	[ax2_x, ax2_y, ax2_w, ax2_h] = [0.3,0.72,0.6,0.15] ### last one controls hight of the dendrogram
	ax2_x = axr_x + axr_w + width_between_axr_axm
	ax2_y = ax1_y + ax1_h + height_between_ax1_axc + axc_h + height_between_axc_ax2
	ax2_w = axc_w

	# axcb - placement of the color legend
	[axcb_x, axcb_y, axcb_w, axcb_h] = [0.07,0.88,0.18,0.09]

	## compute pdist for rows:
	d1 = dist.pdist(data, metric=row_pdist)
	D1 = dist.squareform(d1)
	ax1 = fig.add_axes([ax1_x, ax1_y, ax1_w, ax1_h], frame_on=True)
	Y1 = sch.linkage(D1, method=row_linkage, metric=row_pdist)
	Z1 = sch.dendrogram(Y1, orientation='right')
	idx1 = Z1['leaves']
	ax1.set_xticks([])
	ax1.set_yticks([])

	## compute pdist for cols
	d2 = dist.pdist(data.T, metric=col_pdist)
	D2 = dist.squareform(d2)
	ax2 = fig.add_axes([ax2_x, ax2_y, ax2_w, ax2_h], frame_on=True)
	Y2 = sch.linkage(D2, method=col_linkage, metric=col_pdist)
	Z2 = sch.dendrogram(Y2)
	idx2 = Z2['leaves']
	ax2.set_xticks([])
	ax2.set_yticks([])

	## plot color legend
	display_range = float(display_range)
	### Scale the max and min colors so that 0 is white/black
	vmin=data.min()
	vmax=data.max()
	vmax = max([vmax,abs(vmin)])
	vmin = vmax*-1
	norm = mpl.colors.Normalize(-display_range, display_range)
	axcb = fig.add_axes([axcb_x, axcb_y, axcb_w, axcb_h], frame_on=False)
	step = display_range/3
	bounds = np.arange(-display_range, display_range+step, step)
	cb = mpl.colorbar.ColorbarBase(axcb, cmap=cmap, norm=norm, 
		orientation='horizontal', ticks=bounds, spacing='proportional', extend='both')
	axcb.set_title("colorkey")

	## plot heatmap
	axm = fig.add_axes([axm_x, axm_y, axm_w, axm_h])
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
		if len(row_labels) < 100: ### Don't visualize gene associations when more than 100 rows
			axm.text(data.shape[1]-0.5, i, '  '+row_labels[idx1[i]])
		new_row_header.append(row_labels[idx1[i]])
	for i in range(data.shape[1]):
		if len(col_labels) < 100:
			axm.text(i, -0.9, ' '+ col_labels[idx2[i]], rotation=270, verticalalignment="top") # rotation could also be degrees
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
	
	cmap_group = mpl.colors.ListedColormap(['r', 'g', 'b', 'y', 'w', 'k', 'm'])

	axr = fig.add_axes([axr_x, axr_y, axr_w, axr_h])
	new_row_group = np.array([d_row_group[row_groups[idx1[i]]] for i in range(data.shape[0])])
	new_row_group.shape = (len(idx1), 1)
	axr.matshow(new_row_group, aspect='auto', origin='lower', cmap=cmap_group)
	axr.set_xticks([])
	axr.set_yticks([])

	axc = fig.add_axes([axc_x, axc_y, axc_w, axc_h])
	new_col_group = np.array([d_col_group[col_groups[idx2[i]]] for i in range(data.shape[1])])
	new_col_group.shape = (1, len(idx2))	
	axc.matshow(new_col_group, aspect='auto', origin='lower', cmap=cmap_group)
	axc.set_xticks([])
	axc.set_yticks([])


	plt.show()


## test:

data = np.arange(0,14,1).reshape(2,7)

clustergram(data=np.random.rand(4,3), row_labels=['a','c','e','d'], col_labels=['1','2','3'],
			row_groups=['A','B','C','C'], col_groups=['1','1','2'],
			row_linkage='average', col_linkage='average', 
			row_pdist='euclidean', col_pdist='euclidean',
			standardize=3, log=False, colormap='redbluecmap',
			display_range=3)
