#include "oneflow/core/graph/boxing/sub_task_graph_builder_context.h"

namespace oneflow {

SubTskGphBuilderCtx::SubTskGphBuilderCtx(TaskGraph* task_graph) : task_graph_(task_graph) {}

TaskGraph* SubTskGphBuilderCtx::task_graph() { return task_graph_; }

TaskNode* SubTskGphBuilderCtx::GetProxyNode(TaskNode* src_node, int64_t src_mem_zone_id,
                                            int64_t dst_machine_id, int64_t dst_mem_zone_id) {
  const auto key = std::make_pair(dst_machine_id, dst_mem_zone_id);
  if (node2proxies_.find(src_node) != node2proxies_.cend()
      && node2proxies_.at(src_node).find(key) != node2proxies_.at(src_node).cend()) {
    return node2proxies_.at(src_node).at(key);
  } else {
    if (dst_machine_id == src_node->machine_id() && dst_mem_zone_id == src_mem_zone_id) {
      node2proxies_[src_node][key] = src_node;
      return src_node;
    } else if (Global<IDMgr>::Get()->IsGpuMemZone(dst_mem_zone_id)) {
      TaskNode* proxy_on_dst_host = GetProxyNode(src_node, src_mem_zone_id, dst_machine_id,
                                                 Global<IDMgr>::Get()->CpuMemZoneId());
      CopyHdTaskNode* copy_task = task_graph()->NewNode<CopyHdTaskNode>();
      copy_task->Init(CopyHdOpConf::H2D, proxy_on_dst_host->machine_id(),
                      Global<IDMgr>::Get()->GetGpuPhyIdFromMemZoneId(dst_mem_zone_id));
      Connect<TaskNode>(proxy_on_dst_host, task_graph()->NewEdge(), copy_task);
      node2proxies_[src_node][key] = copy_task;
      return copy_task;
    } else if (Global<IDMgr>::Get()->IsCpuMemZone(dst_mem_zone_id)) {
      if (src_node->machine_id() == dst_machine_id) {
        if (Global<IDMgr>::Get()->IsGpuMemZone(src_mem_zone_id)) {
          CopyHdTaskNode* copy_task = task_graph()->NewNode<CopyHdTaskNode>();
          copy_task->Init(CopyHdOpConf::D2H, src_node->machine_id(),
                          Global<IDMgr>::Get()->GetGpuPhyIdFromMemZoneId(src_mem_zone_id));
          Connect<TaskNode>(src_node, task_graph()->NewEdge(), copy_task);
          node2proxies_[src_node][key] = copy_task;
          return copy_task;
        } else {
          UNIMPLEMENTED();
        }
      } else {
        TaskNode* proxy_on_src_host =
            GetProxyNode(src_node, src_mem_zone_id, src_node->machine_id(),
                         Global<IDMgr>::Get()->CpuMemZoneId());
        CopyCommNetTaskNode* copy_comm_net_task = task_graph()->NewNode<CopyCommNetTaskNode>();
        copy_comm_net_task->Init(dst_machine_id, proxy_on_src_host->machine_id());
        Connect<TaskNode>(proxy_on_src_host, task_graph()->NewEdge(), copy_comm_net_task);
        node2proxies_[src_node][key] = copy_comm_net_task;
        return copy_comm_net_task;
      }
    } else {
      UNIMPLEMENTED();
    }
  }
}

}  // namespace oneflow