
# -*- coding: utf-8 -*-
#生成半月数据

import numpy as np
import matplotlib.pyplot as plt
 
 
def halfmoon(rad, width, d, n_samp): 
    '''生成半月数据
    @param  rad:    半径
    @param  width:  宽度
    @param  d:      距离
    @param  n_samp: 数量
    ''' 
    if n_samp%2 != 0:  
        n_samp += 1  
    
    data = np.zeros((3,n_samp))
      
    aa = np.random.random((2,int(n_samp/2)))  
    radius = (rad-width/2) + width*aa[0,:] 
    theta = np.pi*aa[1,:]        
      
    x     = radius*np.cos(theta)  
    y     = radius*np.sin(theta)  
    label = np.ones((1,len(x)))         # label for Class 1  
      
    x1    = radius*np.cos(-theta) + rad  
    y1    = radius*np.sin(-theta) - d  
    label1= -1*np.ones((1,len(x1)))     # label for Class 2  
     
    data[0,:]=np.concatenate([x,x1])
    data[1,:]=np.concatenate([y,y1])
    data[2,:]=np.concatenate([label,label1],axis=1)
    
    return data
 
def halfmoon_shuffle(rad, width, d, n_samp): 
     
    data = halfmoon(rad, width, d, n_samp)      
    shuffle_seq = np.random.permutation(np.arange(n_samp))  
    data_shuffle = data[:,shuffle_seq]
    
    return data_shuffle
 
 
if __name__ == "__main__":
    dataNum = 1000
    data = halfmoon(10,5,-2,dataNum)
    pos_data = data[:,0: int(dataNum/2)]
    neg_data = data[:, int(dataNum/2):dataNum]
    
    np.savetxt('halfmoon.txt', data.T,fmt='%4f',delimiter=',')
    
    plt.figure()
    plt.scatter(pos_data[0,:],pos_data[1,:],c="b",s=10)
    plt.scatter(neg_data[0,:],neg_data[1,:],c="r",s=10)
    plt.savefig('../imgs/moon_nonlinear.png')
