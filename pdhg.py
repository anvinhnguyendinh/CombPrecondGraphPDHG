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

one_rows[1] *= -1
one_cols[1] *= -1
K1  = spdiags(one_rows, [0, 1], rows-1, rows).toarray() * W
K2  = spdiags(one_cols, [0, 1], cols-1, cols).toarray() * W

max_iters = 100000
gamma = 0.04 #
tol = 1e-7

t = 10.0 * 2 # 20, 2000
s = 0.1

u  = np.zeros(img.shape, dtype=np.float)
v1 = np.zeros(img.shape, dtype=np.float)
v2 = np.zeros(img.shape, dtype=np.float)
p1 = np.zeros([rows-1, cols, channel], dtype=np.float)
p2 = np.zeros([rows, cols-1, channel], dtype=np.float)
p  = np.zeros([2 * rows, cols, channel], dtype=np.float)

def kK(r, c):
	return (2 - np.cos(np.pi * (r - 1) / r) - np.cos(np.pi * (c - 1) / c)) / (1 - np.cos(np.pi / max(r, c)))

def apply_K(mat_u):
	p[:rows-1,    :   ] = np.tensordot(K1, mat_u, ([1], [0]))
	p[rows:  , :cols-1] = np.transpose(np.tensordot(mat_u, K2, ([1], [1])), (0, 2, 1))
	return p
def apply_KT(mat_p1, mat_p2):
	return np.tensordot(K1, mat_p1, ([0], [0])), np.transpose(np.tensordot(mat_p2, K2, ([1], [0])), (0, 2, 1))

def primal(mat_u):
	return 0.5 * np.sum((mat_u - img) ** 2, axis=(0, 1)) + np.sum(np.abs(apply_K(mat_u)), axis=(0, 1))
def dual(mat_p1, mat_p2, kp=None):
	if kp is None:
		kp1, kp2 = apply_KT(mat_p1, mat_p2)
	else:
		kp1, kp2 = kp
	skp = kp1 + kp2
	return np.sum(0.5 * (skp ** 2) - img * skp, axis=(0, 1)), kp1, kp2


def total(mat_f):
	last, lenf = mat_f[-1], len(mat_f)
	lambdas = [(- W + last, W + last)]
	slopePoints = deque([-1, - W + last, 0, W + last, -1])

	for i in range(2, lenf):
		nega, posi = 0, 0
		value, old_lambd = - W - mat_f[-i], 0
		while True:
			slope  = slopePoints.popleft()
			try:
				lambd = slopePoints.popleft()
				value += (slope + i) * (lambd - old_lambd)
				if value >= - W:
					nega = lambd - (value + W) / (slope + i)
					if value > - W:
						slopePoints.extendleft([lambd, slope])
					slopePoints.extendleft([nega, - i])
					break
			except:
				nega = old_lambd + (- W - value) / (slope + i)
				slopePoints.extendleft([slope, nega, - i])
				break
			old_lambd = lambd
		value, old_lambd =   W - mat_f[-i], 0
		while True:
			slope  = slopePoints.pop    ()
			try:
				lambd  = slopePoints.pop    ()
				value += (slope + i) * (lambd - old_lambd)
				if value <= W:
					posi = lambd - (value - W) / (slope + i)
					if value < W:
						slopePoints.extend    ([lambd, slope])
					slopePoints.extend    ([posi, - i])
					break
			except:
				posi = old_lambd - (  W - value) / (slope + i)
				slopePoints.extend    ([slope, posi, - i])
				break
			old_lambd = lambd
		lambdas.append((nega, posi))
	
	mat_v = [0] * lenf
	value, old_lambd = - W - mat_f[ 0], 0
	while True:
		slope  = slopePoints.popleft()
		try:
			lambd = slopePoints.popleft()
			value += (slope + lenf) * (lambd - old_lambd)
			if value >= 0:
				mat_v[0] = lambd - value / (slope + lenf)
				break
		except:
			mat_v[0] = old_lambd - value / (slope + lenf)
			break
		old_lambd = lambd

	temp = mat_v[0]
	for i in range(1, lenf):
		nega, posi = lambdas[-i]
		mat_v[i] = (temp <= nega) * (nega - temp) + (temp >= posi) * (posi - temp) + temp
		temp = mat_v[i]
	return mat_v


def total_eye(mat_f):
	return (mat_f > 1) * (mat_f - 1) + (mat_f < -1) * (mat_f + 1)


def show(mat):
	plt.imshow(mat.astype(np.uint8)[:, :, ::-1])
	plt.show()

en_primal = primal(u)
en_dual, kp1, kp2 = dual(p1, p2)
gap_zero = en_primal + en_dual

coordis, tripoints = [], []
transformers = [PCA(n_components=i) for i in range(4)]

for it in range(max_iters):
	u_prev = u
	u = (s * u_prev + img - kp1 - kp2) / (1 + s)

	theta = 1 / np.sqrt(1 + 2 * gamma * (1 / s))
	s /= theta
	t *= theta
	u_bar = u + theta * (u - u_prev)
	# u_bar = 2 * u - u_prev

	# dual update
	u_bar_t = u_bar / t
	f1 = - kp1 - u_bar_t #
	f2 = - kp2 - u_bar_t

	for ch in range(channel): #
		for c in range(cols):
			v1[:, c, ch] = total(f1[:, c, ch])
		for r in range(rows):
			v2[r, :, ch] = total(f2[r, :, ch])
	# print(total(range(1, 4, 1)))

	kp1 = v1 - f1 #
	kp2 = v2 - f2
	x1 = - t * v1
	x2 = - t * v2
	
	en_primal = primal(u)
	en_dual, _, _ = dual(p1, p2, (kp1, kp2)) #
	gap = np.max((en_primal + en_dual) / gap_zero)

	if (it+1) % 10 == 0:
		coordis.append(u[:, :, 0])
		tripoints.append(en_primal[0])
	# cv2.imwrite('./png/%03d.png' % (it+1), u[:, :, 0:1].astype(np.uint8))
	# cv2.imwrite('./png/%03d_x0.png' % (it+1), u .astype(np.uint8))
	# cv2.imwrite('./png/%03d_x1.png' % (it+1), x1.astype(np.uint8))
	# cv2.imwrite('./png/%03d_x2.png' % (it+1), x2.astype(np.uint8))

	if (it+1) % 10 == 0 or gap < tol: #
		print('%d iterations: duality gap: %.12f\n' % (it+1, gap))
	if gap < tol:
		break

result = u

data = np.reshape(np.array(coordis), (len(coordis), -1))
bipoints  = np.concatenate((transformers[2].fit_transform(data).T, np.array([tripoints])), axis=0)
tripoints = transformers[3].fit_transform(data).T
# with open('PCA_2.obj', 'wb') as file:
# 	pickle.dump(transformers[2], file)
# with open('PCA_3.obj', 'wb') as file:
# 	pickle.dump(transformers[3], file)

fig = plt.figure()
ax = Axes3D(fig)
colors = cm.rainbow(np.linspace(0, 1, tripoints.shape[1]))
tripoints /= np.max(np.abs(tripoints), axis=1, keepdims=True)
ax.scatter(tripoints[0], tripoints[1], tripoints[2], color=colors)
plt.show()

fig2 = plt.figure()
ax = Axes3D(fig2)
colors = cm.rainbow(np.linspace(0, 1, bipoints.shape[1]))
bipoints /= np.max(np.abs(bipoints), axis=1, keepdims=True)
ax.scatter(bipoints[0], bipoints[1], bipoints[2], color=colors)
plt.show()

print(gap_zero, np.mean(x1+x2, axis=(0, 1)) / 2, gap)
# show(x1)