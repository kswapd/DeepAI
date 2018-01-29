import numpy as np
import tensorflow as tf
def main():
     feature_columns = [tf.feature_column.numeric_column("x", shape=[2])]
     #classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[10,5], n_classes=20,model_dir="../tmp/add_model")
     #classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,hidden_units=[10,5], n_classes=20,model_dir="../tmp/classify_add_model")
     classifier = tf.estimator.DNNRegressor(feature_columns=feature_columns,hidden_units=[10,5], model_dir="../tmp/regressor_add_model")
     #classifier = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns,hidden_units=[10,5], model_dir="../tmp/iris_model")
     npx = np.array([[0,0]],dtype=np.float32)
     npy = np.array([0],dtype=np.int)
     for i in range(0,9):
         for j in range(0,9):
             npx = np.append(npx,[[i,j]],axis=0)
             npy = np.append(npy,[i+j], axis=0)
     print("value:{}, {}".format(npx,npy))
     x = tf.constant(npx)
     y = tf.constant(npy)
     train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(npx)},
      y=np.array(npy),
      num_epochs=None,
      shuffle=True)

     test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(npx)},
      y=np.array(npy),
      num_epochs=1,
      shuffle=False)

     new_samples = np.array(
      [[6.4, 3.2],
       [5.8, 3.1]], dtype=np.float32)

     predict_input_fn = tf.estimator.inputs.numpy_input_fn(
     x={"x": new_samples},
     num_epochs=1,
     shuffle=False)
     classifier.train(input_fn=train_input_fn, steps=2000)
     predictions = list(classifier.predict(input_fn=predict_input_fn))
     print("New Samples, Class Predictions: {}\n".format(predictions))
     #predicted_classes = [p["classes"] for p in predictions]
     #print("New Samples, Class Predictions: {}\n".format(predicted_classes))
if __name__ == "__main__":
    main()
