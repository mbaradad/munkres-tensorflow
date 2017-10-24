#include <assert.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"
#include <iostream>
#include "munkres.h"

using std::vector;
using namespace tensorflow;
using shape_inference::Shape;
using shape_inference::Dimension;
using shape_inference::DimensionHandle;
using shape_inference::ShapeHandle;

REGISTER_OP("Hungarian")
    .Input("cost_matrix: float32")
    .Output("assignments: int32")
    //TODO: add shape function
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      //Same shape as input, droping last dimension, as an index is returned for each assignment
        ShapeHandle input = c->input(0);

        if (!c->RankKnown(input)) {
          // If we do not have the rank of the input, we don't know the output shape.
          c->set_output(0, c->UnknownShape());
          return Status::OK();
        }

        const int32 input_rank = c->Rank(input);
        std::vector< DimensionHandle> dims;

        for (int i = 0; i < input_rank - 1; ++i) {
            dims.emplace_back(c->Dim(input, i));
        }

        c->set_output(0, c->MakeShape(dims));

        return Status::OK();
    });

//TODO: add error handling for inf or input with overflow
class HungarianOp : public OpKernel {
 public:
  explicit HungarianOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& costs_tensor = context->input(0);
    auto costs = costs_tensor.tensor<float, 3>();

    // Create an output tensor
    Tensor* assignments_tensor = NULL;
    vector<int64> cost_shape;
    for (int i = 0; i < costs_tensor.shape().dims(); ++i) {
      cost_shape.push_back(costs_tensor.shape().dim_size(i));
    }
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({cost_shape[0], cost_shape[1]}),
                                                     &assignments_tensor));
    auto assignments_output = assignments_tensor->matrix<int>();

    const int batch_size = cost_shape[0];
    auto shard = [&costs, &cost_shape, &assignments_output](int64 start, int64 limit) {
      for (int n = start; n < limit; ++n) {
        Matrix<float> matrix(cost_shape[1], cost_shape[2]);
        for (int i = 0; i < cost_shape[1]; ++i) {
          for (int j = 0; j < cost_shape[2]; ++j) {
            matrix(i,j) =  costs(n, i, j);
          }
        }
        Munkres<float> munk = Munkres<float>();
        munk.solve(matrix);

        for (int i = 0; i < cost_shape[1]; ++i) {
          bool assigned = false;
          for (int j = 0; j < cost_shape[2]; ++j){
            if(matrix(i,j) == 0){
              assigned = true;
              assignments_output(n, i) = j;
            }
            if(!assigned) assignments_output(n, i) = -1;
          }
        }
      }
    };

    // This is just a very crude approximation
    const int64 single_cost = 10000 * cost_shape[1] * cost_shape[1] * cost_shape[2];

    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads->num_threads, worker_threads->workers, batch_size, single_cost, shard);
  }
};

REGISTER_KERNEL_BUILDER(Name("Hungarian").Device(DEVICE_CPU), HungarianOp);
