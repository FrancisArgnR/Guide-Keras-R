# Install-Keras-R-Ubuntu

### 1st step: Install devtools

Devtools is necesary because it allow us to install and packages from GitHub.

- Install system dependencies for devtools (in console): 

  _sudo apt-get install build-essential libcurl4-gnutls-dev libxml2-dev libssl-dev_
  
- Install devtools package (in R): 

  _install.packages('devtools')_
  
### 2nd step: Install Keras

- Install keras from github repository (in R):

  _devtools::install_github("rstudio/keras")_
  
- To make sure Keras is installed (in R):

  _packageVersion("keras")_
  
### 3rd step: Install TensorFlow

- Install TensorFlow (in R):

  _install_tensorflow()_ #for cpu#
  
  _install_tensorflow(gpu = T)_ #for nvdidia gpu#
  
- To make sure TensorFlow is installed (in R):

  _packageVersion("tensorflow")_

### 2nd-3rd in one step: Install Keras and TensorFlow simultaneously:

- Install keras from github repository (in R):

  _devtools::install_github("rstudio/keras")_

- Install system dependencies for TensorFlow (in console):

  _sudo apt-get install python-pip python-virtualenv_
  
- Install Keras and TensorFlow (in R):

  _install_keras()_



  
### References

https://keras.rstudio.com/

https://medium.com/towards-data-science/how-to-implement-deep-learning-in-r-using-keras-and-tensorflow-82d135ae4889

https://www.digitalocean.com/community/tutorials/how-to-install-r-packages-using-devtools-on-ubuntu-16-04
