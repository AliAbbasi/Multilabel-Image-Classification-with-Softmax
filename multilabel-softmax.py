# This is an example of Multi-label Softmax Classifier

# The dataset include two labeled images of objects and shapes

# The dataset consist of 8 category trivial objects include: helmet, kettle, joystick, keyboard, mouse, stapler, barrel and mug
# and small shapes inside each image include: triangle, square and circle 

# The first labels shows the object category and the second label shows the shape inside each image.

# Create by Ali Abbasi, Apr-2017 contact me if you have ant question or suggestion
# email: abbasi.ali.tab@gmail.com
# github: https://github.com/AliAbbasi
# Linkedin: https://www.linkedin.com/in/ali-abbasi-6b5512b7/

#======================================================================================================================================================================================================================

import sys , random
import tensorflow                          as     tf
import numpy                               as     np 
from   random                              import randint
from   PIL                                 import Image
from   numpy                               import array

#======================================================================================================================================================================================================================

data     = np.load('32x32-two-labeled-images.npz')
trX, trY = data[ 'datax' ], data[ 'datay' ]   

# shuffling the arrays
shuffling = list(zip(trX, trY))
random.shuffle(shuffling)              
trX, trY = zip(*shuffling)


trX = np.asarray(trX)
trY = np.asarray(trY)

teX, teY = trX [ 18000:  ], trY [ 18000:  ]  # testset
trX, trY = trX [ :18000  ], trY [ :18000  ]  # trainset

trX = trX.reshape( -1, 32, 32, 1) 
teX = teX.reshape( -1, 1024     )
teY = teY.reshape( -1, 16       )

#======================================================================================================================================================================================================================

class ConvNet( object ):

    def paramsFun(self): 
        params_w = {'wLyr1': tf.Variable(tf.random_normal([ 3, 3, 1,  self.lyr1FilterNo_                        ])),
                    'wLyr2': tf.Variable(tf.random_normal([ 3, 3,     self.lyr1FilterNo_ , self.lyr2FilterNo_   ])),
                    'wLyr3': tf.Variable(tf.random_normal([ 3, 3,     self.lyr2FilterNo_ , self.lyr3FilterNo_   ])),
                    'wFCh':  tf.Variable(tf.random_normal([ 4* 4*     self.lyr3FilterNo_ , self.fcHidLyrSize_   ])),   
                    'wOut':  tf.Variable(tf.random_normal([           self.fcHidLyrSize_ , self.outLyrSize_     ]))}
                
        params_b = {'bLyr1': tf.Variable(tf.random_normal([           self.lyr1FilterNo_                        ])),
                    'bLyr2': tf.Variable(tf.random_normal([           self.lyr2FilterNo_                        ])),
                    'bLyr3': tf.Variable(tf.random_normal([           self.lyr3FilterNo_                        ])),
                    'bFCh':  tf.Variable(tf.random_normal([           self.fcHidLyrSize_                        ])),
                    'bOut':  tf.Variable(tf.random_normal([           self.outLyrSize_                          ]))}
        return params_w,params_b

    #======================================================================================================================================================================================================================

    def scoreFun(self):

        def conv2d(x, W, b, strides=1):
            x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
            x = tf.nn.bias_add(x, b)
            return x

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  

        def maxpool2d(x, k=2):
            return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
        
        self.x_ = tf.reshape(x, shape = [-1,32,32,1])
        
        # 1)  
        convLyr_1_conv = conv2d (self.x_, self.params_w_['wLyr1'], self.params_b_['bLyr1'])
        convLyr_1_relu = tf.nn.relu(convLyr_1_conv) 
        convLyr_1_pool = maxpool2d(convLyr_1_relu, k=2)
        
        # 2)
        convLyr_2_conv = conv2d(convLyr_1_pool, self.params_w_['wLyr2'], self.params_b_['bLyr2'])
        convLyr_2_relu = tf.nn.relu(convLyr_2_conv)
        convLyr_2_pool = maxpool2d(convLyr_2_relu, k=2)

        # 3)
        convLyr_3_conv = conv2d(convLyr_2_pool, self.params_w_['wLyr3'], self.params_b_['bLyr3'])
        convLyr_3_relu = tf.nn.relu(convLyr_3_conv)
        convLyr_3_pool = maxpool2d(convLyr_3_relu, k=2)
        
        # 4) Fully Connected
        fcLyr_1 = tf.reshape(convLyr_3_pool, [-1,self.params_w_['wFCh'].get_shape().as_list()[0]])
        fcLyr_1 = tf.add(tf.matmul(fcLyr_1, self.params_w_['wFCh']), self.params_b_['bFCh'])
        fcLyr_1 = tf.nn.relu(fcLyr_1)
        fcLyr_1 = tf.nn.dropout(fcLyr_1, self.keepProb_)
        
        netOut = tf.add(tf.matmul(fcLyr_1, self.params_w_['wOut']), self.params_b_['bOut'])
        
        # packing the results   
        PackShow = [convLyr_1_conv,convLyr_1_relu,convLyr_2_conv,convLyr_2_relu,convLyr_2_pool, convLyr_3_conv,convLyr_3_relu]
        
        return netOut, PackShow
        
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 

    def costFun(self):
    
        score_split = tf.split( 1, 2, self.score_ )
        label_split = tf.split( 1, 2, self.y_     ) 
        total = 0.0
        for i in range ( len(score_split)  ): 
            total += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( score_split[i] , label_split[i] ))    
        return total
        
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   

    def updateFun(self):
        return tf.train.AdamOptimizer(learning_rate = self.lr_).minimize(self.cost_)

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def perfFun(self): 
    
        score_split = tf.split( 1, 2, self.score_ )
        label_split = tf.split( 1, 2, self.y_     ) 
        
        correct_pred1  = tf.equal(tf.argmax(score_split[0],1), tf.argmax(label_split[0],1))  
        correct_pred2  = tf.equal(tf.argmax(score_split[1],1), tf.argmax(label_split[1],1))  
        
        return correct_pred1 , correct_pred2
        
   #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    

    def __init__(self,x,y,lr,lyr1FilterNo,lyr2FilterNo,lyr3FilterNo,fcHidLyrSize,inLyrSize,outLyrSize, keepProb):
        
        self.x_            = x
        self.y_            = y
        self.lr_           = lr
        self.inLyrSize     = inLyrSize
        self.outLyrSize_   = outLyrSize
        self.lyr1FilterNo_ = lyr1FilterNo
        self.lyr2FilterNo_ = lyr2FilterNo
        self.lyr3FilterNo_ = lyr3FilterNo
        self.fcHidLyrSize_ = fcHidLyrSize
        self.keepProb_     = keepProb

        [self.params_w_, self.params_b_] = ConvNet.paramsFun(self) # initialization and packing the parameters
        self.score_, self.PackShow_      = ConvNet.scoreFun (self) # Computing the score function
        self.cost_                       = ConvNet.costFun  (self) # Computing the cost function
        self.update_                     = ConvNet.updateFun(self) # Computing the update function
        self.perf_1,self.perf_2          = ConvNet.perfFun  (self) # performance

#======================================================================================================================================================================================================================

if __name__ == '__main__':
    
    lyr1FilterNo = 32 
    lyr2FilterNo = 64 
    lyr3FilterNo = 128 

    fcHidLyrSize = 1024
    inLyrSize    = 32 * 32
    outLyrSize   = 16
    lr           = 0.001
    batch_size   = 300
    
    dropout      = 0.5
    x            = tf.placeholder(tf.float32, [None, inLyrSize ])
    y            = tf.placeholder(tf.float32, [None, outLyrSize])
    keepProb     = tf.placeholder(tf.float32) 
    
    ConvNet_class = ConvNet(x,y,lr,lyr1FilterNo,lyr2FilterNo,lyr3FilterNo,fcHidLyrSize,inLyrSize,outLyrSize, keepProb)
    initVar = tf.global_variables_initializer()
    
    
    with tf.Session() as sess:
        sess.run(initVar)
        print("-------------------------------------------------------------")   
       
        index = 0
        trainplot = []
        testplot  = []
        steps     = []

        for batch_i in range(10000):
            trData_i, trLabel_i = [], []
            
            trData_i .append( trX[ index : index + batch_size ] )
            trLabel_i.append( trY[ index : index + batch_size ] )
            index += batch_size
            if index > ( len(trX) - batch_size+1 ):
                index = 0

            trData_i  = np.reshape( trData_i , ( -1, 32 * 32 ) )
            trLabel_i = np.reshape( trLabel_i, ( -1, 16     ) )

            
            update_i, PackShow, wLyr1_i, wLyr2_i, wLyr3_i = sess.run([ConvNet_class.update_, ConvNet_class.PackShow_,
                        ConvNet_class.params_w_['wLyr1'],ConvNet_class.params_w_['wLyr2'] ,
                        ConvNet_class.params_w_['wLyr3']], 
                        feed_dict = { x:trData_i, y:trLabel_i, keepProb:dropout} )
              
            if batch_i%10 == 0: 
            
                cost_tr = sess.run(ConvNet_class.cost_, feed_dict={x: trData_i,    y: trLabel_i,   keepProb: 1.})
                cost_te = sess.run(ConvNet_class.cost_, feed_dict={x: teX[:3000],  y: teY[:3000],  keepProb: 1.})
                
                # test accuracy
                accu1, accu2 = sess.run([ConvNet_class.perf_1 , ConvNet_class.perf_2] , feed_dict={x: teX[:3000],  y: teY[:3000],  keepProb: 1.})
                numOfposit = 0.0
                for tt in range(accu1.shape[0]):
                    if accu1[tt] == accu2[tt] and accu1[tt] == True:
                        numOfposit += 1
                test_accu = numOfposit / accu1.shape[0]
                
                # train accuracy
                accu1, accu2 = sess.run([ConvNet_class.perf_1 , ConvNet_class.perf_2] , feed_dict={x: trData_i,    y: trLabel_i,   keepProb: 1.})
                numOfposit = 0.0
                for tt in range(accu1.shape[0]):
                    if accu1[tt] == accu2[tt] and accu1[tt] == True:
                        numOfposit += 1
                train_accu = numOfposit / accu1.shape[0]
                
                print("%4d, cost_tr: %4.2g , cost_te: %4.2g , trainAccu: %4.2g , testAccu: %4.2g "% ( batch_i , cost_tr , cost_te , train_accu , test_accu ) )
                
                
                
