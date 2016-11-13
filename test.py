import tensorflow as tf
import numpy as np
import os

class MunkresTestSimple(tf.test.TestCase):
  def testMunkres(self):
    tests = get_tests()
    for in_hung, out_hung in tests:
      with self.test_session():
        hungarian_module = tf.load_op_library('hungarian.so')

        result = hungarian_module.hungarian(in_hung).eval()
        self.assertAllEqual(result, out_hung)

  #def testMunkresMultiplePerBatch(self):
        #  return 0


def get_tests():
  tests = list()
  for i in range(0, len(os.listdir('test_files'))/2):
    in_data = np.expand_dims(np.genfromtxt('test_files/test' + str(i) + '.in', delimiter=','), 0)
    out_data = np.expand_dims(np.genfromtxt('test_files/test' + str(i) + '.out', delimiter=','), 0)
    tests.append((in_data, out_data))
  return tests