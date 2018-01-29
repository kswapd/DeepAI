import numpy as np
import tensorflow as tf
features = ["sum1", "sum2"]
tf.logging.set_verbosity(tf.logging.INFO)

def main():
    feature_columns = [tf.contrib.layers.real_valued_column(k) for k  in features]
     #classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[10,5], n_classes=20,model_dir="../tmp/iris_model")
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns,hidden_units=[10,5], model_dir="../tmp/reg_add_model")
    def get_train_inputs():
         npy = np.array([0],dtype=np.float)
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
         print("value:{}".format(label_x))
         y = tf.constant(npy)
         return label_x, y
    regressor.fit(input_fn=get_train_inputs, steps=2000)

    # Score accuracy
    '''def test_samples():
        label_x = {};
        label_x["sum1"] = tf.constant(np.array([2.2, 3.1,5],dtype=np.float))
        label_x["sum2"] = tf.constant(np.array([3.3,4.6, 7],dtype=np.float))
        y = tf.constant(np.array([5.5, 7.7, 12],dtype=np.float))
        return label_x,y
    ev = regressor.evaluate(test_samples, steps=1)
    loss_score = ev["loss"]
    print("Loss: {0:f}".format(loss_score))
    '''




    def new_samples():
        return {"sum1":[2.2, 3.1,5], "sum2":[3.3,4.6, 7]}
    predictions = list(regressor.predict(input_fn=new_samples))
    print("New Samples, Class Predictions:    {}\n".format(predictions))
if __name__ == "__main__":
    main()
