import random
import torch
import torch.utils
import torch.utils.data
from utils import computeMaxSpanningTreesBatch, pairwise_distances

class ParametrizedSoftMax(torch.nn.Module):
    def __init__(self,alpha,dim:int) -> None:        
        torch.nn.Module.__init__(self)
        self.alpha = alpha
        self.dim = dim

    def forward(self,X):
        X = X + torch.min(X)
        X = torch.clamp(X, min=0.0, max=10.0)        
        eX = torch.exp(-self.alpha*X)
        sm = eX/eX.sum(dim=self.dim,keepdim=True)
        return sm


class Fuzzy_ElpiGraph(torch.nn.Module):
    def __init__(self,
                 dimension,
                 number_of_nodes,
                 alphaKmeans=5.0,
                 alphaFuzzyGraph=5.0,
                 lmda=0.01,mu=0.1,
                 y=None,A=None,
                 computeMST=True,
                 intrinsicOptimization=True,
                 numberIterations = 50,
                 intrinsicLearningRate = 0.05) -> None:
        torch.nn.Module.__init__(self)
        if y is None:
            self.y = torch.nn.Parameter(torch.rand(number_of_nodes,dimension,requires_grad=True))
        else:
            self.y = y
        if A is None:
            self.A = torch.eye(number_of_nodes)
        else:
            self.A = A
        self._alphaKmeans = alphaKmeans
        self._alphaFuzzyGraph = alphaFuzzyGraph
        self._lmda = lmda
        self._mu = mu
        self._doComputeMST = computeMST
        self._doOntrinsicOptimization = intrinsicOptimization
        self._intrinsicLearningRate = intrinsicLearningRate
        self._numberIterations = numberIterations

    def _computeFuzzyKmeansMSE(self,X):
        sftmx = ParametrizedSoftMax(alpha=self._alphaKmeans,dim=1)
        dst = pairwise_distances(X,self.y)
        SA = sftmx(dst)
        lossMSE = (dst*SA).sum(dim=1).sum()/X.shape[0]
        return lossMSE

    def _computeStretchingPenalty(self):
        # sum of squared edge lengths
        edge_weights = torch.triu(self.A, diagonal=1)
        diff = self.y.unsqueeze(1) - self.y.unsqueeze(0)
        squared_diff = torch.sum(diff ** 2, dim=-1)
        sum_squared_length = torch.sum(squared_diff * edge_weights)    
        return sum_squared_length

    def _computeHarmonicPenalty(self):
        # harmonicity term
        degrees = torch.sum(self.A, dim=1)
        mask = (degrees > 1).float()
        neighbor_sum = torch.matmul(self.A, self.y)
        avg_neighbour_position = torch.zeros_like(self.y)
        avg_neighbour_position[mask.bool()] = neighbor_sum[mask.bool()] / degrees[mask.bool()].unsqueeze(1)
        diff = self.y - avg_neighbour_position
        squared_distance_harmonic = torch.sum(diff ** 2, dim=1)
        squared_distance_harmonic = (squared_distance_harmonic * mask).sum()
        return squared_distance_harmonic
    
    def _computeFuzzyGraph(self):
        sftmx = ParametrizedSoftMax(alpha=self._alphaFuzzyGraph,dim=1)
        ydst = pairwise_distances(self.y,self.y)
        vdiag = torch.tensor([100]*ydst.shape[0])
        mask = torch.diag(torch.ones_like(vdiag))
        ydst = mask*torch.diag(vdiag) + (1. - mask)*ydst
        A = sftmx(ydst)
        self.A = (A+A.T)/2

    def _computeMST(self):
        A2 = torch.zeros(2,self.A.shape[0],self.A.shape[1])
        A2[0,:,:] = self.A
        A2[1,:,:] = self.A
        A_tree = computeMaxSpanningTreesBatch(A2,2)[0,:,:]
        self.A = A_tree

    def _getLoss(self,X):
        return self._computeFuzzyKmeansMSE(X) + self._lmda*self._computeStretchingPenalty() + self._mu*self._computeHarmonicPenalty()

    def _intrinsic_optimization(self,X,verbose=False):
        optim = torch.optim.Adam([self.y], self._intrinsicLearningRate)
        numiter = self._numberIterations
        if verbose:
            print(f'{numiter=}')
        for i in range(numiter):
            if verbose:
                print(f'step={i+1},MSE={self._computeFuzzyKmeansMSE(X)},Stretch={self._lmda*self._computeStretchingPenalty()},Harmonic={self._mu*self._computeHarmonicPenalty()}')
            optim.zero_grad()
            loss = self._getLoss(X)
            loss.backward(retain_graph=True)
            optim.step()

    def _initNodePositions(self,X):
        lst = list(range(X.shape[0]))
        perm = torch.randperm(X.size(0))
        idx = perm[:self.y.shape[0]]
        samples = X[idx]        
        self.y =  torch.nn.Parameter(torch.tensor(samples.detach().numpy(),requires_grad=True))


    def forward(self, X):
        self._computeFuzzyGraph()
        if self._doComputeMST:
            self._computeMST()
        if self._doOntrinsicOptimization:
            self._intrinsic_optimization(X)
        return self.y, self._getLoss(X)