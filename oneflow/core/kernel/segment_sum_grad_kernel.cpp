#include "oneflow/core/kernel/segment_sum_grad_kernel.h"
#include "oneflow/core/kernel/segment_kernel_util.h"

namespace oneflow {

const PbMessage& SegmentSumGradKernel::GetCustomizedOpConf() const {
  return this->op_conf().segment_sum_grad_conf();
}

void SegmentSumGradKernel::ForwardDataContent(const KernelCtx& ctx,
                                    std::function<Blob*(const std::string&)> BnInOp2Blob) const{
  const Blob* out_diff = BnInOp2Blob("out_diff");
  const Blob* segment_ids = BnInOp2Blob("segment_ids");
  Blob* in_diff = BnInOp2Blob("in_diff");
  // do some real work here
  SegmentKernelUtil<device_type, T>::SegmentSumBackward(ctx.device_ctx, out_diff, segment_ids,
                                                        in_diff);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSegmentSumGradConf, SegmentSumGradKernel, FLOATING_DATA_TYPE_SEQ);

} // namespace oneflow
