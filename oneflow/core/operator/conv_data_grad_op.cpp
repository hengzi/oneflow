#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/operator/conv_data_grad_op.h"
#include "oneflow/core/device/cudnn_conv_util.h"

namespace oneflow {

const PbMessage& ConvDataGradOp::GetCustomizedConf() const {
  return op_conf().conv_data_grad_conf();
}

void ConvDataGradOp::InitFromOpConf() {
  CHECK(op_conf().has_conv_data_grad_conf());
  EnrollInputBn("dy", false);
  EnrollInputBn("filter", false);
  EnrollInputBn("x_like", false)->set_use_header_only(true);
  EnrollOutputBn("dx", false);
  if (DevIsGpuAndEnableCudnn()) {
    EnrollTmpBn("buf");
  } else {
    UNIMPLEMENTED();
  }
}

Maybe<void> ConvDataGradOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  const ConvDataGradOpConf& conf = this->op_conf().conv_data_grad_conf();
  const BlobDesc* dy = GetBlobDesc4BnInOp("dy");
  const BlobDesc* x_like = GetBlobDesc4BnInOp("x_like");
  const int32_t num_spatial_dims = conf.conv_conf().num_spatial_dims();
  CHECK_GE_OR_RETURN(num_spatial_dims, 1);
  CHECK_LE_OR_RETURN(num_spatial_dims, 3);
  CHECK_EQ_OR_RETURN(dy->shape().NumAxes(), num_spatial_dims + 2);
  CHECK_EQ_OR_RETURN(x_like->shape().NumAxes(), num_spatial_dims + 2);
  CHECK_EQ_OR_RETURN(x_like->data_type(), dy->data_type());
  BlobDesc* dx = GetBlobDesc4BnInOp("dx");
  dx->CopyMetaFrom(*x_like);
  return Maybe<void>::Ok();
}

Maybe<void> ConvDataGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  const ConvDataGradOpConf& conf = this->op_conf().conv_data_grad_conf();
  const ConvConf& conv_conf = conf.conv_conf();
  const BlobDesc* dy = GetBlobDesc4BnInOp("dy");
  const BlobDesc* filter = GetBlobDesc4BnInOp("filter");
  const BlobDesc* x_like = GetBlobDesc4BnInOp("x_like");
  const int32_t num_spatial_dims = conf.conv_conf().num_spatial_dims();
  CHECK_GE_OR_RETURN(num_spatial_dims, 1);
  CHECK_LE_OR_RETURN(num_spatial_dims, 3);
  CHECK_EQ_OR_RETURN(dy->shape().NumAxes(), num_spatial_dims + 2);
  CHECK_EQ_OR_RETURN(x_like->shape().NumAxes(), num_spatial_dims + 2);
  CHECK_EQ_OR_RETURN(x_like->data_type(), dy->data_type());
  BlobDesc* dx = GetBlobDesc4BnInOp("dx");
  dx->CopyMetaFrom(*x_like);
  if (DevIsGpuAndEnableCudnn()) {
#ifdef WITH_CUDA
    size_t workspace_size = cudnn_buf_limit_byte();
    if (!dx->is_dynamic()) {
      CudnnConvArgs args(conv_conf, dx->data_type(), ShapeView(dx->shape()), filter->data_type(),
                         ShapeView(filter->shape()), dy->data_type(), ShapeView(dy->shape()),
                         conv_conf.data_format(), workspace_size,
                         job_desc().job_conf().cudnn_conv_heuristic_search_algo(),
                         job_desc().job_conf().cudnn_conv_use_deterministic_algo_only(),
                         job_desc().job_conf().cudnn_conv_enable_pseudo_half());
      using perf_t = cudnnConvolutionBwdDataAlgoPerf_t;
      using algo_t = cudnnConvolutionBwdDataAlgo_t;
      perf_t algo_perf;
      if (this->job_desc().job_conf().has_cudnn_conv_force_bwd_data_algo()) {
        algo_perf = GetCudnnConvAlgorithmPerference<perf_t>(
            &args,
            static_cast<algo_t>(this->job_desc().job_conf().cudnn_conv_force_bwd_data_algo()));
      } else {
        algo_perf = FindCudnnConvAlgorithm<perf_t>(&args);
      }
      CHECK_EQ_OR_RETURN(algo_perf.status, CUDNN_STATUS_SUCCESS)
          << "op (" << op_conf().name()
          << ") find algorithm perference failed. algo: " << algo_perf.algo;
      CHECK_LE_OR_RETURN(algo_perf.memory, workspace_size)
          << "op (" << op_conf().name() << ") find algorithm " << algo_perf.algo << ", need memory "
          << algo_perf.memory << ", but cudnn_buf_limit_byte is " << workspace_size;
      workspace_size = algo_perf.memory;
    }
    workspace_size = std::max(size_t(1), workspace_size);
    BlobDesc* cudnn_buf = GetBlobDesc4BnInOp("buf");
    cudnn_buf->set_data_type(DataType::kChar);
    cudnn_buf->mut_shape() = Shape({static_cast<int64_t>(workspace_size)});
#else
    UNIMPLEMENTED_THEN_RETURN();
#endif
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
  return Maybe<void>::Ok();
}  // namespace oneflow

Maybe<void> ConvDataGradOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  CHECK_OR_RETURN(*BatchAxis4BnInOp("dy") == *BatchAxis4BnInOp("x_like"));
  CHECK_OR_RETURN(BatchAxis4BnInOp("filter")->has_value() == false);
  *BatchAxis4BnInOp("dx") = *BatchAxis4BnInOp("dy");
  return Maybe<void>::Ok();
}

Maybe<void> ConvDataGradOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split("dy", 0)
      .Broadcast("filter")
      .Split("x_like", 0)
      .Split("dx", 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kConvDataGradConf, ConvDataGradOp);

}  // namespace oneflow