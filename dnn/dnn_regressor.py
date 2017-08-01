import numpy as np
import tensorflow as tf
features = ["sum1", "sum2"]

def main():
    feature_columns = [tf.contrib.layers.real_valued_column(k) for k  in features]
     #classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[10,5], n_classes=20,model_dir="../tmp/iris_model")
    classifier = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns,hidden_units=[10,5], model_dir="../tmp/reg_model")
    def get_train_inputs():
         #x = tf.constant(np.array([[1,2,3,4,5,6,7,8,9],[2,1,3,4,5,6,7,2,3]],dtype=np.float32))
         npy = np.array([0],dtype=np.int)
         label_x = {};
         sum1 = np.array([0],dtype=np.float)
         sum2 = np.array([0],dtype=np.float)
         for i in range(0,9):
             for j in range(0,9):
                 sum1 = np.append(sum1,[i],axis=0)
                 sum2 = np.append(sum2,[j],axis=0)
                 npy = np.append(npy,[i+j], axis=0)
         label_x["sum1"] = tf.constant(sum1)
         label_x["sum2"] = tf.constant(sum2) 
         #print("value:{}, {}".format(sum1,sum2))
         print("value:{}".format(label_x))
         y = tf.constant(npy)
         return label_x, y
    classifier.fit(input_fn=get_train_inputs, steps=2000)
    def new_samples():
        return {"sum1":[2.2, 3.1,5], "sum2":[3.3,4.6, 7]}
    predictions = list(classifier.predict(input_fn=new_samples))
    print("New Samples, Class Predictions:    {}\n".format(predictions))
if __name__ == "__main__":
    main()
