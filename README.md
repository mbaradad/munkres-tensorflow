# munkres-tensorflow
Port of Hungarian method/Munkres algorithm C++ implementation to tensorflow 1.0 interface 
## Functionality
This is a port of https://github.com/saebyn/munkres-cpp to Tensorflow interface.

This version allows K arbitrary cost matrices (NxM) as input, and outputs K vectors of N elements, corresponding to the perfect bipartite matching that minimizes the summatory of the selected elements. 

The input tensor should be of rank 3, with dimensions [batch_size, N, M], where each element is a cost matrix.
The output tensor will be of rank 2, with dimensions [batch_size, N], corresponding to the  assignment.

## Usage
Follow the steps in https://www.tensorflow.org/versions/r0.10/how_tos/adding_an_op/index.html to use user defined functions. 
For tensorflow binary installation, it consists in:

### 1-Compiling the function
```bash

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -shared hungarian.cc -o hungarian.so -fPIC -I $TF_INC
```

### 2-Loading the function when building the graph (for other methods, refer to link above)
```python

import tensorflow as tf
hungarian_module = tf.load_op_library('./hungarian.so')
with tf.Session(''):
  print hungarian_module.hungarian([[[1, 2], [3, 4]]]).eval()

#Prints
[[0 1]]
```
## Testing
```bash
cd test
python -m unittest discover

```

## TODO's
1. Add shape inference
2. Improve testing, and test batches bigger than 1.
