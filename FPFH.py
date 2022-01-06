import numpy as np
import torch
import os


class FPFH(object):

    def __init__(self, e, div, nneighbors, rad):
        """Pass in parameters """
        self._e = e
        self._div = div
        self._nneighbors = nneighbors
        self._radius = rad

        self._error_list = []
        self._Rlist = []
        self._tlist = []

    def getNeighbors(self, pq, pc):
        """Get k nearest neighbors of the query point pq from pc, within the radius

        :pq: TODO
        :pc: TODO
        :returns: TODO

        """
        k = self._nneighbors
        ind = torch.arange(pc.shape[0])
        dist = torch.norm(pc - pq, dim=1)
        mask = dist < self._radius
        dist = dist[mask]
        ind = ind[mask]
        dist[dist.min(0)[1]] = 1000
        neighbors = []
        for i in range(k):
            t = dist.min(0)[1]
            neighbors.append(ind[t])
            dist[t] = 1000

        return neighbors

    def step(self, si, fi):
        """Helper function for calc_pfh_hist. Depends on selection of div

        :si: TODO
        :fi: TODO
        :returns: TODO

        """
        result = 0
        if self._div == 2:
            if fi < si[0]:
                result = 0
            else:
                result = 1
        elif self._div == 3:
            if fi < si[0]:
                result = 0
            elif fi >= si[0] and fi < si[1]:
                result = 1
            else:
                result = 2
        elif self._div == 4:
            if fi < si[0]:
                result = 0
            elif fi >= si[0] and fi < si[1]:
                result = 1
            elif fi >= si[1] and fi < si[2]:
                result = 2
            else:
                result = 3
        elif self._div == 5:
            if fi < si[0]:
                result = 0
            elif fi >= si[0] and fi < si[1]:
                result = 1
            elif fi >= si[1] and fi < si[2]:
                result = 2
            elif fi >= si[2] and fi < si[3]:
                result = 3
            else:
                result = 4

        return result

    def calc_thresholds(self):
        """
        :returns: 3x(div-1) array where each row is a feature's thresholds
        """
        delta = 2. / self._div
        s1 = torch.tensor([-1 + i * delta for i in range(1, self._div)])

        delta = 2. / self._div
        s3 = torch.tensor([-1 + i * delta for i in range(1, self._div)])

        delta = (np.pi) / self._div
        s4 = torch.tensor([-np.pi / 2 + i * delta for i in range(1, self._div)])

        s = torch.tensor([s1, s3, s4]).unsqueeze(1)
        return s

    def calc_pfh_hist(self, f):
        """Calculate histogram and bin edges.

        :f: feature vector of f1,f3,f4 (Nx3)
        :returns:
            pfh_hist - array of length div^3, represents number of samples per bin
            bin_edges - range(0, 1, 2, ..., (div^3+1))
        """
        # preallocate array sizes, create bin_edges
        pfh_hist, bin_edges = torch.zeros(self._div ** 3), torch.arange(0, self._div ** 3 + 1)

        # find the division thresholds for the histogram
        s = self.calc_thresholds()

        # Loop for every row in f from 0 to N
        for j in range(0, f.shape[0]):
            # calculate the bin index to increment
            index = 0
            for i in range(1, 4):
                index += self.step(s[i - 1, :], f[j, i - 1]) * (self._div ** (i - 1))

            # Increment histogram at that index
            pfh_hist[index] += 1

        return pfh_hist, bin_edges

    def calc_normals(self, pc):

        normals = []
        ind_of_neighbors = []
        N = int(pc.shape[0])
        for i in range(N):
            # Get the indices of neighbors, it is a list of tuples (dist, indx)
            indN = self.getNeighbors(pc[i], pc)
            # indN = list((neigh.kneighbors(pc[i].reshape(1, -1), return_distance=False)).flatten())
            # indN.pop(0)

            # Breakout just the indices
            indN = [indN[i] for i in range(len(indN))]  # <- old code
            ind_of_neighbors.append(indN)

            # PCA
            # X = utils.convert_pc_to_matrix(pc)[:, indN]
            X = pc[indN, :]
            X = X - torch.mean(X, dim=0)
            cov = torch.matmul(X.T, X) / (len(indN))
            _, _, Vt = torch.svd(cov)
            normal = Vt[2, :]

            # Re-orient normal vectors
            if torch.matmul(normal, -1. * (pc[i])) < 0:
                normal = -1. * normal
            normals.append(normal)

        return normals, ind_of_neighbors

    def calcHistArray(self, pc, norm, indNeigh):
        """Overriding base PFH to FPFH"""

        print("\tCalculating histograms fast method \n")
        N = len(pc)
        histArray = torch.zeros((N, self._div ** 3))
        distArray = torch.zeros((self._nneighbors))
        distList = []
        for i in range(N):
            u = torch.as_tensor(norm[i].T).squeeze()

            features = torch.zeros((len(indNeigh[i]), 3))
            for j in range(len(indNeigh[i])):
                pi = pc[i]
                pj = pc[indNeigh[i][j]]
                if torch.arccos(torch.dot(norm[i], pj - pi)) <= torch.arccos(torch.dot(norm[j], pi - pj)):
                    ps = pi
                    pt = pj
                    ns = torch.as_tensor(norm[i]).squeeze()
                    nt = torch.as_tensor(norm[indNeigh[i][j]]).squeeze()
                else:
                    ps = pj
                    pt = pi
                    ns = torch.as_tensor(norm[indNeigh[i][j]]).squeeze()
                    nt = torch.as_tensor(norm[i]).squeeze()

                u = ns
                difV = pt - ps
                dist = torch.norm(difV)
                difV = difV / dist
                difV = torch.as_tensor(difV).squeeze()
                v = torch.cross(difV, u)
                w = torch.cross(u, v)

                alpha = torch.dot(v, nt)
                phi = torch.dot(u, difV)
                theta = torch.arctan(torch.dot(w, nt) / torch.dot(u, nt))

                features[j, 0] = alpha
                features[j, 1] = phi
                features[j, 2] = theta
                distArray[j] = dist

            distList.append(distArray)
            pfh_hist, bin_edges = self.calc_pfh_hist(features)
            histArray[i, :] = pfh_hist / (len(indNeigh[i]))

        fast_histArray = torch.zeros_like(histArray)
        for i in range(N):
            k = len(indNeigh[i])
            spfh_sum = torch.zeros_like(histArray[i])
            for j in range(k):
                spfh_sum += histArray[indNeigh[i][j]] * (1 / distList[i][j])

            fast_histArray[i, :] = histArray[i, :] + (1 / k) * spfh_sum

        return fast_histArray

    def solve(self, pc):

        pc = pc[:, :3]
        normS,indS = self.calc_normals(pc)
        histS = self.calcHistArray(pc, normS, indS)

        return histS


# if __name__ == '__main__':
#
#     source_path = "/home/autocars/Desktop/kitti/training"
#     label_path = os.path.join(source_path, "label_2")
#     file_list = os.listdir(label_path)
#     file_name = file_list[0].split('.')[0]
#     points, _, _ = get_data(source_path, file_name)
#
#     et = 0.1
#     div = 2
#     nneighbors = 8
#     rad = 100#0.03
#     fpfh = FPFH(et, div, nneighbors, rad)   # Fast PFH
#     hist = fpfh.solve(points)
#     print(hist)