# ClinicApp Models

The code in this repo converts logistic regression models originally created in R to study Ebola diagnosis and prognosis into [TensorFlow models](https://www.tensorflow.org/), and save them in [TensorFlow Lite](https://www.tensorflow.org/lite) format to load into ClinicApp.

RCS terms are included into the TensorFlow model using a [custom Keras layer](https://www.tensorflow.org/tutorials/customization/custom_layers) where the RCS formula is implemented, and then incorporated as an input layer. See the ebola-pediatric-prognosis notebook for details.