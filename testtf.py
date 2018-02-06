import tensorflow as tf
global_step = tf.train.get_or_create_global_step()
  
a=tf.constant([  
        [[1.0,2.0,3.0,4.0],  
        [5.0,6.0,7.0,8.0],  
        [9,10,11,12],  
        [13,14,15,16]],  
        [[17,18,19,20],  
         [21,22,23,24],  
         [25,26,27,28],  
         [29,30,31,32]]  
    ])  
  
a=tf.reshape(a,[1,4,4,2])  
  
pooling=tf.nn.max_pool(a,[1,2,2,1],[1,1,1,1],padding='VALID')  
with tf.Session() as sess:  
    print("image:")  
    image=sess.run(a)  
    print (image)  
    print("reslut:")  
    result=sess.run(pooling)  
    print (result)  

print ('{:}'.format(global_step))
