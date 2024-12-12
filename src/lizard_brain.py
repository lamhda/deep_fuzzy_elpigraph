import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def lizard_brain(number_of_branches = 10,
                 dimension = 10, 
                 epsilon = 0.005, 
                 add_noise = 0.001, 
                 min_branch_points = 50, 
                 k_forknngraph = 8, 
                 make_knn_graph = False,
                 show_fig = False,
                 noise_type = 'laplacian',
                 normalize_to_interval = True
                ):

    x0 = np.zeros((1,dimension)) 
    i1 = 0 
    i2 = 1 
    branch = [] 
    while len(branch)<min_branch_points:
        x0[:,i1] = np.random.random() 
        x0[:,i2] = np.random.random() 
        branch = make_branch(x0,i1,i2,epsilon) 

    data = branch 
    irx = [1]*len(data)

    k = 0 
    while k<=number_of_branches:
        n = np.floor(len(branch)/2) 
        #x0 = branch(n,:) 
        x0 = data[[int(np.floor(np.random.random()*len(data)))]]

        i1 = int(np.floor(np.random.random()*dimension)) 
        i2 = int(np.floor(np.random.random()*dimension)) 

        while (i2==i1):
            i2 = int(np.floor(np.random.random()*dimension)) 
        
        #print('Dim (%i,%i)'%(i1,i2))

        newbranch = make_branch(x0,i1,i2,epsilon) 
        n1 = len(data) 
        n2 = len(newbranch) 

        if n2>min_branch_points-1:
            data = np.concatenate((data,newbranch)) 
            irx.extend([k+2]*n2)
            branch = newbranch 
            k = k+1 
        # plot(branch(:,1),branch(:,2),'ko')  hold on 
        #  plot([x0(:,1) x0(:,1)+v1(:,1)/20],[x0(:,2) x0(:,2)+v1(:,2)/20],'b-') 
        #  plot([x0(:,1) x0(:,1)+v2(:,1)/20],[x0(:,2) x0(:,2)+v2(:,2)/20],'b-') 

    if add_noise>0:
        if noise_type=='uniform':
            data = data + np.random.random((len(data),data.shape[1]))*add_noise 
        if noise_type=='gaussian':
            data = data + np.random.normal(0,1,(len(data),data.shape[1]))*add_noise
        if noise_type=='laplacian':
            data = data + np.random.laplace(0,1,(len(data),data.shape[1]))*add_noise
        
    if normalize_to_interval:
        min_val = np.min(data)
        max_val = np.max(data)
        data = (data - min_val) / (max_val - min_val)*2.0-1.0


    pca = PCA()
    u = pca.fit_transform(data)
    v = pca.components_.T
    s = pca.explained_variance_
    if show_fig:
        unlab = np.unique(irx)
        cols = ['k','r','g','b','m','c','y','grey','purple','palegreen']
        for i,lab in enumerate(unlab):
            col = cols[i%10]
            plt.plot(u[irx==lab,0],u[irx==lab,1],'k.',color=col,markersize=1,linewidth=1)
        plt.axis('equal')
        plt.show()

    if make_knn_graph:

        knngraph, _ = knnsearch(data,data,k_forknngraph) 
        with open('knn1.sif','w')  as fid:
            for i in range(len(knngraph)):
                for k in range(knngraph.shape[1]):
                    fid.write('%i\tna\t%i\n'%(knngraph[i,0],knngraph[i,k])) 
                    plt.plot([u[knngraph[i,0],0], u[knngraph[i,k],0]],[u[knngraph[i,0],1], u[knngraph[i,k],1]],'b--') 

        plt.show()
    return [data,np.array(irx),v,u,s]

def make_branch(x0,i1,i2,epsilon):
    dimension = x0.shape[1] 
    v1 = np.zeros((1,dimension)) 
    v2 = np.zeros((1,dimension)) 
    v1[:,i1] = np.random.random()-0.5  
    v1[:,i2] = np.random.random()-0.5  
    v1 = v1/np.linalg.norm(v1) 
    v2[:,i1] = -v1[:,i2]  
    v2[:,i2] = v1[:,i1] 
    return parabolic_branch(x0,v1,v2,epsilon,dimension) 

def parabolic_branch(x0,v1,v2,epsilon,dimension):
    x = np.zeros((1,dimension)) 
    t = epsilon/1000 
    i = 0
    irx1 = np.where(v1!=0)[1] 
    irx2 = np.where(v2!=0)[1] 
    while 1:
        xn = x0+t*v1+t*t*v2 
        if (np.max(xn[:,irx1])<1) and (np.min(xn[:,irx2])>0):
            if i == 0:
                x = xn
                i = i+1 
                t = t+epsilon
            else:
                x = np.concatenate((x,xn))
                i = i+1 
                t = t+epsilon 
        else:
            break
    return x