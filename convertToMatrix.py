import numpy       as     np
import scipy.misc  
import glob

#------------------------------------------------------------------------------------------------------------

datax = [] # train data 
datay = [] # train data label

for imgName in glob.glob('*.png'):
    x = scipy.misc.imread( imgName, flatten=False, mode='L' ) 
    y = np.zeros((16))  
    
    if   imgName[1] == 'a':
        y[ int(str(imgName[0])) ] = 1
        y[ 8                    ] = 1 
    elif imgName[1] == 'b': 
        y[ int(str(imgName[0])) ] = 1
        y[ 10                   ] = 1 
    else:    
        y[ int(str(imgName[0])) ] = 1
        y[ 9                    ] = 1
    
    datax.append(x)
    datay.append(y)


np.savez( 'mydata', datax = datax , datay = datay )

#------------------------------------------------------------------------------------------------------------
