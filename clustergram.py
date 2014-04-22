"""
A clustergram function similar to MATLAB clustergram()

Author: Zichen Wang
Created on 4/7/2014

Major enhancement: enables group labels for rows and columns, which can be useful to 
directly visualize whether the hierarchical clustering outcome agree with inherent 
catagories of samples.

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
			row_groups=None, col_groups=None, cluster=True,
			row_linkage='average', col_linkage='average', 
			row_pdist='euclidean', col_pdist='euclidean',
			standardize=3, log=False, colormap='redbluecmap',
			display_range=3, figsize=12, figname=None, colorkey='colorkey'):
	"""
	Parameters:
	----------
	data: a numpy array or numpy matrix
	row_labels: a list of strings corresponding to the rows in data
	col_labels: a list of strings corresponding to the columns in data
	row_groups: a list of strings used as the group labels for the rows in data
	col_groups: a list of strings used as the group labels for the columns in data
	cluster: boolean variable specifying whether to perform hierarchical clustering (True) or not (False)
	row_linkage: linkage method used for rows
	col_linkage: linkage method used for columns 
		options = ['average','single','complete','weighted','centroid','median','ward']
	row_pdist: pdist metric used for rows
	col_pdist: pdist metric used for columns
		options = ['euclidean','minkowski','cityblock','seuclidean','sqeuclidean',
		'cosine','correlation','hamming','jaccard','chebyshev','canberra','braycurtis',
		'mahalanobis','wminkowski']
	standardize: specifying the dimension for standardizing the values in data
		options =  {1: 'standardize along the columns of data',
					2: 'standardize along the rows of data',
					3: 'do not standardize the data'}
	log: boolean variable specifying whether to perform log transform for the data
	colormap: options = ['redbluecmap', 'redgreencmap']
	display_range: specifies the display range of values of data,
		if number is specified:
			display_range is zero centered
		elif a list or tuple of two numbers:
			display_range is the exact same range with the input
	figsize: specifies the size of the figure
	figname: figure name (format of figure should be specified, e.g. .png, .pdf),
		if specified, figure will be saved instead of being shown
	colorkey: specifies the name of the colorkey to display

	Example:
	----------
	from clustergram import clustergram
	clustergram(data=np.random.randn(3,3),row_labels=['a','b','c'],
		col_labels=['1','2','3'], row_groups=['A','A','B'],
		col_groups=['group1','group1','group2'])

	"""
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
	elif colormap == 'grey':
		cmap = plt.cm.Greys

	### Configure the Matplotlib figure size
	default_window_hight = figsize
	default_window_width = figsize
	fig = plt.figure(figsize=(default_window_width, default_window_hight)) ### could use m,n to scale here
	color_bar_w = 0.01 
	group_bar_w = 0.01
	heatmap_w = 0.7
	heatmap_h = 0.5
	dendrogram_l = 0.15
	color_legend_w = 0.18
	color_legend_h = 0.09
	margin = 0.01
	fig_margin_l = 0.05
	fig_margin_b = 0.10
	## calculate positions for all elements
	# ax1, placement of dendrogram 1, on the left of the heatmap
	rect1 = [fig_margin_l, fig_margin_b, dendrogram_l, heatmap_h]
	# axr, placement of row side colorbar
	rectr = [fig_margin_l + dendrogram_l, fig_margin_b, color_bar_w, heatmap_h]
	# axc, placement of column side colorbar
	rectc = [fig_margin_l + dendrogram_l + group_bar_w + margin, heatmap_h + fig_margin_b + margin, heatmap_w, color_bar_w]
	# axm, placement of heatmap
	rectm = [fig_margin_l + dendrogram_l + group_bar_w + margin, fig_margin_b, heatmap_w, heatmap_h]
	# ax2, placement of dendrogram 2, on the top of the heatmap
	rect2 = [fig_margin_l + dendrogram_l + group_bar_w + margin, fig_margin_b + heatmap_h + group_bar_w, heatmap_w, dendrogram_l] ### last one controls hight of the dendrogram
	# axcb - placement of the color legend
	rectcb = [0.05,0.85,0.15,0.08]

	## plot color legend
	if type(display_range) == int or type(display_range) == float:
		display_range = float(display_range)
		norm = mpl.colors.Normalize(-display_range, display_range)
		step = display_range/2
		bounds = np.arange(-display_range, display_range+step, step)
	else:
		if len(display_range) == 2:
			norm = mpl.colors.Normalize(display_range[0], display_range[1])
			step = (display_range[1]-display_range[0])/4.
			bounds = np.arange(display_range[0], display_range[1]+step,step)
	axcb = fig.add_axes(rectcb, frame_on=False)
	cb = mpl.colorbar.ColorbarBase(axcb, cmap=cmap, norm=norm, 
		orientation='horizontal', ticks=bounds, spacing='proportional', extend='both')
	axcb.set_title(colorkey)

	if cluster: ## perform hierarchical clustering for rows and cols
		## compute pdist for rows:
		d1 = dist.pdist(data, metric=row_pdist)
		D1 = dist.squareform(d1)
		ax1 = fig.add_axes(rect1, frame_on=False)
		Y1 = sch.linkage(D1, method=row_linkage, metric=row_pdist)
		Z1 = sch.dendrogram(Y1, orientation='right')
		idx1 = Z1['leaves']
		ax1.set_xticks([])
		ax1.set_yticks([])

		## compute pdist for cols
		d2 = dist.pdist(data.T, metric=col_pdist)
		D2 = dist.squareform(d2)
		ax2 = fig.add_axes(rect2, frame_on=False)
		Y2 = sch.linkage(D2, method=col_linkage, metric=col_pdist)
		Z2 = sch.dendrogram(Y2)
		idx2 = Z2['leaves']
		ax2.set_xticks([])
		ax2.set_yticks([])

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
		if row_labels is not None:
			if len(row_labels) < 150:
				for i in range(data.shape[0]):
					axm.text(data.shape[1]-0.5, i, '  '+row_labels[idx1[i]])
					new_row_header.append(row_labels[idx1[i]])
		if col_labels is not None:
			if len(col_labels) < 150:
				for i in range(data.shape[1]):
					axm.text(i, -0.8, ' '+ col_labels[idx2[i]], rotation=270, verticalalignment="top") # rotation could also be degrees
					new_column_header.append(col_labels[idx2[i]])

		## plot group colors
		# numerize group names
		if row_groups != None:
			uniq_row_groups = list(set(row_groups))
			d_row_group = {} 
			for i, group_name in enumerate(uniq_row_groups):
				d_row_group[group_name] = i

			colors_row_groups = []
			for i in range(len(d_row_group)):
				colors_row_groups.append(np.random.rand(3,1)) 
			cmap_row_groups = mpl.colors.ListedColormap(colors_row_groups) ## make color lists into cmap for matshow

			## row group color label:
			axr = fig.add_axes(rectr)
			new_row_group = np.array([d_row_group[row_groups[idx1[i]]] for i in range(data.shape[0])])
			new_row_group.shape = (len(idx1), 1)
			rmat = axr.matshow(new_row_group, aspect='auto', origin='lower', cmap=cmap_row_groups)
			axr.set_xticks([])
			axr.set_yticks([])

			## axglr: placement for row group label legends
			axglr = fig.add_axes([0.8, fig_margin_b, 0.05, 0.3], frame_on=False)
			rcbar = fig.colorbar(rmat, cax=axglr, ticks=range(len(d_row_group)))
			rcbar.set_label('row groups')
			rcbar.set_ticklabels(d_row_group.keys())
			rcbar.update_ticks()

		if col_groups != None:
			uniq_col_groups = list(set(col_groups))
			d_col_group = {} 
			for i, group_name in enumerate(uniq_col_groups):
				d_col_group[group_name] = i
			
			## config group colors and cmaps
			colors_col_groups = []
			for i in range(len(d_col_group)):
				colors_col_groups.append(np.random.rand(3,1)) ## a list of random colors
			cmap_col_groups = mpl.colors.ListedColormap(colors_col_groups)

			axc = fig.add_axes(rectc)
			new_col_group = np.array([d_col_group[col_groups[idx2[i]]] for i in range(data.shape[1])])
			new_col_group.shape = (1, len(idx2))	
			cmat = axc.matshow(new_col_group, aspect='auto', origin='lower', cmap=cmap_col_groups)
			axc.set_xticks([])
			axc.set_yticks([])

			## axglc: placement for col group label legends
			axglc = fig.add_axes([0.8, 0.5, 0.05, 0.3], frame_on=False)
			ccbar = fig.colorbar(cmat, cax=axglc, ticks=range(len(d_col_group)))
			ccbar.set_label('column groups')
			ccbar.set_ticklabels(d_col_group.keys())
			ccbar.update_ticks()
	else: ## not performing hierachical clustering
		## plot heatmap
		axm = fig.add_axes(rectm)
		im = axm.matshow(data, aspect='auto', origin='lower',cmap=cmap, norm=norm)
		axm.set_xticks([])
		axm.set_yticks([])

		## add labels
		if row_labels is not None:
			if len(row_labels) < 150:
				for i in range(data.shape[0]):
					axm.text(data.shape[1]-0.5, i, '  '+row_labels[i])
		if col_labels is not None:
			if len(col_labels) < 150:
				for i in range(data.shape[1]):
					axm.text(i, -0.8, ' '+ col_labels[i], rotation=270, verticalalignment="top") # rotation could also be degrees
	if figname != None:
		plt.savefig(figname)
	else:
		plt.show()

def read_matrix(fn, sep='\t'):
	"""
	a function that helps quickly import data frame from text file
	"""
	with open (fn) as f:
		header = f.next()
		col_labels = header.strip().split(sep)
		row_labels = []
		data = []
		for line in f:
			sl = line.strip().split('\t')
			row_labels.append(sl[0])
			data.append(sl[1:])
		data = np.array(data, dtype=float)
	return data, col_labels, row_labels

