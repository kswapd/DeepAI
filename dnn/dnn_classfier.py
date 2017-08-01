import numpy as np
import tensorflow as tf
def main():
     feature_columns = [tf.contrib.layers.real_valued_column("", dimension=2)]
     classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[10,5], n_classes=20,model_dir="../tmp/iris_model")
     #classifier = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns,hidden_units=[10,5], model_dir="../tmp/iris_model")
     def get_train_inputs():
         #x = tf.constant(np.array([[1,2,3,4,5,6,7,8,9],[2,1,3,4,5,6,7,2,3]],dtype=np.float32))
         npx = np.array([[0,0]],dtype=np.float32)
         npy = np.array([0],dtype=np.int)
         for i in range(0,9):
             for j in range(0,9):
                 npx = np.append(npx,[[i,j]],axis=0)
                 npy = np.append(npy,[i+j], axis=0)

        
         print("value:{}, {}".format(npx,npy))
         #print(npx)
         #print(npy)
         #x = tf.constant(np.array([[1,2],[2,1],[3,3],[4,4],[5,5],[6,6],[7,7],[4,6],[9,3],[6,3],[6,4],[3,2],[3,5],[4,5]],dtype=np.float32))
         #y = tf.constant(np.array([3,3,6,8,10,12,14,10,12,9,10,5,8,9],dtype=np.int))
         x = tf.constant(npx)
         y = tf.constant(npy)
         return x, y
     classifier.fit(input_fn=get_train_inputs, steps=2000)
     def new_samples():
         return np.array([[6.2, 3.1],[3, 1]], dtype=np.float32)
     predictions = list(classifier.predict(input_fn=new_samples))
     print("New Samples, Class Predictions:    {}\n".format(predictions))
if __name__ == "__main__":
    main()
