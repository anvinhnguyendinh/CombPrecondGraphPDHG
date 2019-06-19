import cv2, pickle
import numpy as np
import matplotlib.cm as cm
from collections import deque
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

img = cv2.imread('noisy_input.png').astype(np.float)

rows, cols, channel = img.shape
rowcols = rows * cols
W = 10.0

one_rows = np.ones((2, rows), dtype=np.float)
one_cols = np.ones((2, cols), dtype=np.float)
p = np.zeros([2 * rows, cols, 1], dtype=np.float) #

one_rows[1] *= -1
one_cols[1] *= -1
K1  = spdiags(one_rows, [0, 1], rows-1, rows).toarray() * W
K2  = spdiags(one_cols, [0, 1], cols-1, cols).toarray() * W

print(np.linalg.eig(K1.dot(np.linalg.pinv(K1.T.dot(K1))).dot(K1.T)))

def apply_K(mat_u):
	p[:rows-1,    :   ] = np.tensordot(K1, mat_u, ([1], [0]))
	p[rows:  , :cols-1] = np.transpose(np.tensordot(mat_u, K2, ([1], [1])), (0, 2, 1))
	return p
def primal(mat_u):
	return 0.5 * np.sum((mat_u - img) ** 2, axis=(0, 1)) + np.sum(np.abs(apply_K(mat_u)), axis=(0, 1))

primal_non, primal_pre = [], []
coordis_non, coordis_pre = [], []

for i in range(1,   93,  1):
	img_ = cv2.imread('./png_none/%03d.png' % i).astype(np.float)
	coordis_non.append(img_[:, :, 0])
	primal_non.append(primal(img_[:, :, 0:1])[0])
for i in range(1,   93,  1):
	img_ = cv2.imread(     './png/%03d.png' % i).astype(np.float)
	coordis_pre.append(img_[:, :, 0])
	primal_pre.append(primal(img_[:, :, 0:1])[0])

transformers = [PCA(n_components=i) for i in range(4)]
with open('PCA_2_none.obj', 'rb') as file:
	transformers[2] = pickle.load(file)
with open('PCA_3_none.obj', 'rb') as file:
	transformers[3] = pickle.load(file)

data_non = np.reshape(np.array(coordis_non), (len(coordis_non), -1))
data_pre = np.reshape(np.array(coordis_pre), (len(coordis_pre), -1))
bipoints_non  = np.concatenate((transformers[2].transform(data_non).T, np.array([primal_non])), axis=0)
bipoints_pre  = np.concatenate((transformers[2].transform(data_pre).T, np.array([primal_pre])), axis=0)
tripoints_non = transformers[3].transform(data_non).T
tripoints_pre = transformers[3].transform(data_pre).T

fig = plt.figure()
ax = Axes3D(fig)
colors  = cm.rainbow(np.linspace(0, 0.5, tripoints_non.shape[1]))
colors_ = cm.rainbow(np.linspace(0.5, 1, tripoints_pre.shape[1]))
trinormal = np.max(np.abs(tripoints_non), axis=1, keepdims=True)
# tripoints_non /= trinormal
# tripoints_pre /= trinormal
ax.plot   (tripoints_non[0], tripoints_non[1], tripoints_non[2], 'b-')
ax.plot   (tripoints_pre[0], tripoints_pre[1], tripoints_pre[2], 'g-')
ax.scatter(tripoints_non[0], tripoints_non[1], tripoints_non[2], color=colors)
ax.scatter(tripoints_pre[0], tripoints_pre[1], tripoints_pre[2], color=colors_)
plt.show()

fig2 = plt.figure()
ax = Axes3D(fig2)
colors  = cm.rainbow(np.linspace(0, 0.5, bipoints_non.shape[1]))
colors_ = cm.rainbow(np.linspace(0.5, 1, bipoints_pre.shape[1]))
binormal = np.max(np.abs(bipoints_non), axis=1, keepdims=True)
# bipoints_non /= binormal
# bipoints_pre /= binormal
ax.plot   (bipoints_non[0], bipoints_non[1], bipoints_non[2], 'b-')
ax.plot   (bipoints_pre[0], bipoints_pre[1], bipoints_pre[2], 'g-')
ax.scatter(bipoints_non[0], bipoints_non[1], bipoints_non[2], color=colors)
ax.scatter(bipoints_pre[0], bipoints_pre[1], bipoints_pre[2], color=colors_)
plt.show()