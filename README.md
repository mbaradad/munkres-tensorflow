# munkres-tensorflow
Port of Munkres C++ implementation to tensorflow interface 
## Functionality
This is a port from libhungarian by _Cyrill Stachniss_, based on a previous port by Russell91 https://github.com/Russell91/TensorBox/blob/master/utils/hungarian/hungarian.cc.

This version allows an arbitrary cost matrix (MxM, squared for the moment) as input, and outputs a vector of M elements, corresponding to the perfect bipartite matching that minimizes the sumatory of the selected elements. 

## Usage
Follow the steps in https://www.tensorflow.org/versions/r0.10/how_tos/adding_an_op/index.html to use user defined functions. 
For tensorflow binary installation, it consists in:

1-Compiling the function
```bash

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -shared hungarian.cc -o hungarian.so -fPIC -I $TF_INC
```

1-Loading the function when building the graph (for other methods, refer to link above)
```python

import tensorflow as tf
zero_out_module = tf.load_op_library('zero_out.so')
with tf.Session(''):
  zero_out_module.zero_out([[1, 2], [3, 4]]).eval()

# Prints
array([[1, 0],
       [0, 0]], dtype=int32)
```

## TODO's
1. Add comptibility for rectangular cost matrix
2. Add shape inference
