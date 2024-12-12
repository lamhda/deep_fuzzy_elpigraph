import torch
import numpy as np
import matplotlib.pyplot as plt

def draw_graph2D(X,y,A,nodecolor='r',title='',scale=1.0,show=True,drawX=True):
    yn = y.detach().numpy()
    #sns.scatterplot(x=X[:,0],y=X[:,1],color=sxn)
    if drawX:
        plt.plot(X[:,0],X[:,1],'k.')
    plt.plot(yn[:,0],yn[:,1],nodecolor+'o',markersize=15*scale)
    for ki in range(yn.shape[0]):
        for kj in range(yn.shape[0]):
            plt.plot([yn[ki,0],yn[kj,0]],[yn[ki,1],yn[kj,1]],nodecolor+'-',alpha=A[ki,kj].item(),linewidth=A[ki,kj].item()*scale*10)
    plt.title(title)
    plt.axis('equal')
    if show:
        plt.show()
    
def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist    

def computeMaxSpanningTreesBatch(Q,S):

    nV = Q.shape[1]

    grid = torch.arange(0, (nV)*(nV)).reshape(nV,nV).triu(1)

    triuIdx = grid[grid>0]

    seqS= torch.arange(0,S)

    def extractTriuAndFlatten(A):
        return A.reshape(S,nV*nV)[:, triuIdx]

    inTree = torch.zeros((S, nV, 1), dtype=torch.bool)
    inTree[:,nV-1] = True

    triuBool = (grid>0)
    Q_flat = Q.reshape([S, nV*nV])[:,triuIdx]

    A_trees= torch.zeros(S,nV,nV)

    for step in range(nV-1):

        validEdgeIdx = (inTree != torch.transpose(inTree,1,2)) & triuBool
        drawnEdgeIdx = triuIdx[ (Q_flat * extractTriuAndFlatten(validEdgeIdx)).argmax(1)]

        drawn_j = drawnEdgeIdx % nV
        drawn_i = (drawnEdgeIdx / nV).long()

        inTree[seqS,drawn_i]=True
        inTree[seqS,drawn_j]=True
        A_trees[seqS,drawn_i,drawn_j]= True
        A_trees[seqS,drawn_j,drawn_i]= True
    
    return A_trees

