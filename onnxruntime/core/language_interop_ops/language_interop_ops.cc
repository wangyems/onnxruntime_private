#include "language_interop_ops.h"
#include "core/platform/env.h"
#include "core/session/inference_session.h"
#include "pyop/pyop.h"
#include <google/protobuf/io/zero_copy_stream_impl.h>

namespace onnxruntime {

void InterOpDomainDeleter(OrtCustomOpDomain* domain) {
    if (nullptr != domain) {
        for (auto op: domain->custom_ops_) {
            delete op;
        }
        delete domain;
    }
}

void LoadInterOp(const std::basic_string<ORTCHAR_T>& model_uri, InterOpDomains& domains, const InterOpLogFunc& log_func)
{
    int fd;
    ORT_ENFORCE(Env::Default().FileOpenRd(model_uri, fd).IsOK(), "Failed to read model file");
    google::protobuf::io::FileInputStream f(fd);
    f.SetCloseOnDelete(true);
    ONNX_NAMESPACE::ModelProto model_proto;
    ORT_ENFORCE(model_proto.ParseFromZeroCopyStream(&f), "Failed to parse model proto");
    LoadInterOp(model_proto, domains, log_func);
}

void LoadInterOp(const ONNX_NAMESPACE::ModelProto& model_proto, InterOpDomains& domains, const InterOpLogFunc& log_func)
{
    LoadInterOp(model_proto.graph(), domains, log_func);
}

void LoadInterOp(const ONNX_NAMESPACE::GraphProto& graph_proto, InterOpDomains& domains, const InterOpLogFunc& log_func) {
    for (int i = 0; i < graph_proto.node_size(); ++i) {
        const auto& node_proto = graph_proto.node(i);
        if (node_proto.op_type() == "PyOp") {
            auto pyop_domain = OrtCreateCustomOpDomain(node_proto.domain().c_str());
            ORT_THROW_ON_ERROR(OrtCustomOpDomain_Add(pyop_domain, LoadPyOp(node_proto, log_func)));
            domains.push_back(std::move(std::unique_ptr<OrtCustomOpDomain,decltype(&InterOpDomainDeleter)>(pyop_domain, &InterOpDomainDeleter)));
        } else {
            for (int j = 0; j < node_proto.attribute_size(); ++j) {
                const auto& attr = node_proto.attribute(j);
                if (attr.has_g()) {
                    LoadInterOp(attr.g(), domains, log_func); //load pyop in subgraph
                }
            }//for
        }//else
    }//for
}
}
