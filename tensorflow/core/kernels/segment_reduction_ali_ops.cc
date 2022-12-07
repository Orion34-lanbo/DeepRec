/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <vector>
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/segment_reduction_ali_ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/util.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentReductionMeanOp
    : public SparseSegmentReductionOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentReductionMeanOp(OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T, Index, SegmentId>(
            context, true /*is_mean*/, false /*is_sqrtn*/,
            false /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentReductionMeanWithNumSegmentsOp
    : public SparseSegmentReductionOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentReductionMeanWithNumSegmentsOp(
      OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T, Index, SegmentId>(
            context, true /*is_mean*/, false /*is_sqrtn*/,
            true /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentReductionSqrtNOp
    : public SparseSegmentReductionOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentReductionSqrtNOp(OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T, Index, SegmentId>(
            context, false /*is_mean*/, true /*is_sqrtn*/,
            false /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentReductionSqrtNWithNumSegmentsOp
    : public SparseSegmentReductionOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentReductionSqrtNWithNumSegmentsOp(
      OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T, Index, SegmentId>(
            context, false /*is_mean*/, true /*is_sqrtn*/,
            true /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentReductionSumOp
    : public SparseSegmentReductionOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentReductionSumOp(OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T, Index, SegmentId>(
            context, false /*is_mean*/, false /*is_sqrtn*/,
            false /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentReductionSumWithNumSegmentsOp
    : public SparseSegmentReductionOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentReductionSumWithNumSegmentsOp(
      OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T, Index, SegmentId>(
            context, false /*is_mean*/, false /*is_sqrtn*/,
            true /* has_num_segments */, T(0) /* default_value */) {}
};

#define REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE(type, index_type) \
  REGISTER_CPU_SPARSE_KERNELS(type, index_type, int32)                         \
  REGISTER_CPU_SPARSE_KERNELS(type, index_type, int64)
#define REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(type)       \
  REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE(type, int32) \
  REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE(type, int64)

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type)       \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("SparseSegmentSum")                                                \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<type>("T")                                          \
          .TypeConstraint<index_type>("Tidx")                                 \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),                   \
      SparseSegmentReductionSumOp<CPUDevice, type, index_type,                \
                                  segment_ids_type>);                         \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("SparseSegmentSumWithNumSegments")                                 \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<type>("T")                                          \
          .TypeConstraint<index_type>("Tidx")                                 \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),                   \
      SparseSegmentReductionSumWithNumSegmentsOp<CPUDevice, type, index_type, \
                                                 segment_ids_type>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type)        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("SparseSegmentMean")                                                \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<index_type>("Tidx")                                  \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),                    \
      SparseSegmentReductionMeanOp<CPUDevice, type, index_type,                \
                                   segment_ids_type>);                         \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("SparseSegmentMeanWithNumSegments")                                 \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<index_type>("Tidx")                                  \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),                    \
      SparseSegmentReductionMeanWithNumSegmentsOp<CPUDevice, type, index_type, \
                                                  segment_ids_type>);
TF_CALL_FLOAT_TYPES(REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_CPU_SPARSE_KERNELS

define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type) \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentSqrtN")                                        \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentReductionSqrtNOp<CPUDevice, type, index_type,        \
                                    segment_ids_type>);                 \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentSqrtNWithNumSegments")                         \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentReductionSqrtNWithNumSegmentsOp<                     \
          CPUDevice, type, index_type, segment_ids_type>);
TF_CALL_FLOAT_TYPES(REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_CPU_SPARSE_KERNELS


template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentSumGradOp
    : public SparseSegmentGradOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentSumGradOp(OpKernelConstruction* context)
      : SparseSegmentGradOpBase<Device, T, Index, SegmentId>(
            context, SparseSegmentReductionOperation::kSum) {}
};

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentMeanGradOp
    : public SparseSegmentGradOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentMeanGradOp(OpKernelConstruction* context)
      : SparseSegmentGradOpBase<Device, T, Index, SegmentId>(
            context, SparseSegmentReductionOperation::kMean) {}
};

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentSqrtNGradOp
    : public SparseSegmentGradOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentSqrtNGradOp(OpKernelConstruction* context)
      : SparseSegmentGradOpBase<Device, T, Index, SegmentId>(
            context, SparseSegmentReductionOperation::kSqrtN) {}
};

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type) \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentSumGrad")                                      \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentSumGradOp<CPUDevice, type, index_type, segment_ids_type>);
REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(float);
REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(double);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type) \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentMeanGrad")                                     \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentMeanGradOp<CPUDevice, type, index_type, segment_ids_type>);
REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(float);
REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(double);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type) \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentSqrtNGrad")                                    \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentSqrtNGradOp<CPUDevice, type, index_type,             \
                               segment_ids_type>);
REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(float);
REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(double);
#undef REGISTER_CPU_SPARSE_KERNELS

#undef REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE
#undef REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE

// ____________________________________________________________________________
// Sparse segment reduction ops.

// Same as SegmentReductionOp but takes as input a "sparse" tensor, represented
// by two dense tensors, one containing the data, and the other containing
// indices into the data.
template <typename Device, class T>
class SparseSegmentReductionAliOpBase : public OpKernel {
 public:
  explicit SparseSegmentReductionAliOpBase(OpKernelConstruction* context,
                                          bool is_mean, bool is_sqrtn,
                                          bool has_num_segments,
                                          T default_value)
      : OpKernel(context),
        has_num_segments_(has_num_segments),
        reducer_(is_mean, is_sqrtn, has_num_segments, default_value) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& segment_ids = context->input(2);
    
    Tensor num_segments;
    if (has_num_segments_) {
      num_segments = context->input(3);
    }

    Tensor output;
    reducer_.Reduce(context, input, indices, segment_ids, num_segments,
        context->output_alloc_attr(0), &output);
    context->set_output(0, output);
  }

 private:
  const bool has_num_segments_;
  SparseSegmentReduction<Device, T> reducer_;
};

}  // namespace tensorflow
