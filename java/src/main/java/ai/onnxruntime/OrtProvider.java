/*
 * Copyright (c) 2020, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.util.HashMap;
import java.util.Map;

/** The execution providers available through the Java API. */
public enum OrtProvider {
  CPU("CPUExecutionProvider"),
  CUDA("CUDAExecutionProvider"),
  DNNL("DnnlExecutionProvider"),
  NGRAPH("NGRAPHExecutionProvider"),
  OPEN_VINO("OpenVINOExecutionProvider"),
  NUPHAR("NupharExecutionProvider"),
  VITIS_AI("VitisAIExecutionProvider"),
  TENSOR_RT("TensorrtExecutionProvider"),
  NNAPI("NnapiExecutionProvider"),
  RK_NPU("RknpuExecutionProvider"),
  DIRECT_ML("DmlExecutionProvider"),
  MI_GRAPH_X("MIGraphXExecutionProvider"),
  ACL("ACLExecutionProvider"),
  ARM_NN("ArmNNExecutionProvider");

  private static final Map<String, OrtProvider> valueMap = new HashMap<>(values().length);

  static {
    for (OrtProvider p : OrtProvider.values()) {
      valueMap.put(p.name, p);
    }
  }

  private final String name;

  OrtProvider(String name) {
    this.name = name;
  }

  public static OrtProvider mapFromName(String name) {
    OrtProvider provider = valueMap.get(name);
    if (provider == null) {
      throw new IllegalArgumentException("Unknown execution provider - " + name);
    } else {
      return provider;
    }
  }
}
