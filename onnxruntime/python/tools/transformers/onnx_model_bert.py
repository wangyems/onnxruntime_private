#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

from logging import getLogger
from typing import List
from onnx import ModelProto, TensorProto, helper
from onnx_model import OnnxModel
from fusion_reshape import FusionReshape
from fusion_layernorm import FusionLayerNormalization, FusionLayerNormalizationTF
from fusion_skiplayernorm import FusionSkipLayerNormalization, FusionBiasSkipLayerNormalization
from fusion_embedlayer import FusionEmbedLayerNormalization
from fusion_attention import FusionAttention, AttentionMask, AttentionMaskFormat
from fusion_gelu import FusionGelu
from fusion_fastgelu import FusionFastGelu
from fusion_biasgelu import FusionBiasGelu
from fusion_gelu_approximation import FusionGeluApproximation
from fusion_utils import FusionUtils
from fusion_options import FusionOptions

logger = getLogger(__name__)


class BertOptimizationOptions(FusionOptions):
    """ This class is deprecated
    """
    def __init__(self, model_type):
        logger.warning(f"BertOptimizationOptions is depreciated. Please use FusionOptions instead.")
        super().__init__(model_type)


class BertOnnxModel(OnnxModel):
    def __init__(self, model: ModelProto, num_heads: int = 0, hidden_size: int = 0):
        """Initialize BERT ONNX Model.

        Args:
            model (ModelProto): the ONNX model
            num_heads (int, optional): number of attentioin heads. Defaults to 0, and we will detect the parameter automatically.
            hidden_size (int, optional): hidden dimension. Defaults to 0, and we will detect the parameter automatically.
        """
        assert (num_heads == 0 and hidden_size == 0) or (num_heads > 0 and hidden_size % num_heads == 0)

        super().__init__(model)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.attention_mask = AttentionMask(self)
        self.attention_fusion = FusionAttention(self, self.hidden_size, self.num_heads, self.attention_mask)
        self.utils = FusionUtils(self)

    def fuse_attention(self):
        self.attention_fusion.apply()

    def fuse_gelu(self):
        fusion = FusionGelu(self)
        fusion.apply()
        fusion = FusionFastGelu(self)
        fusion.apply()

    def fuse_bias_gelu(self, is_fastgelu):
        fusion = FusionBiasGelu(self, is_fastgelu)
        fusion.apply()

    def gelu_approximation(self):
        fusion = FusionGeluApproximation(self)
        fusion.apply()

    def fuse_add_bias_skip_layer_norm(self):
        fusion = FusionBiasSkipLayerNormalization(self)
        fusion.apply()

    def fuse_reshape(self):
        fusion = FusionReshape(self)
        fusion.apply()

    def fuse_embed_layer(self):
        fusion = FusionEmbedLayerNormalization(self)
        fusion.apply()

    def fuse_layer_norm(self):
        fusion = FusionLayerNormalization(self)
        fusion.apply()

        fusion = FusionLayerNormalizationTF(self)
        fusion.apply()

    def fuse_skip_layer_norm(self):
        fusion = FusionSkipLayerNormalization(self)
        fusion.apply()

    def get_graph_inputs_from_node_type(self, op_type: str, input_indices: List[int], casted: bool):
        """
        Get graph inputs that feed into node type (like EmbedLayerNormalization or Attention).
        Returns a list of the graph input names based on the filter whether it is casted or not.
        """
        graph_inputs = []

        output_name_to_node = self.output_name_to_node()
        nodes = self.get_nodes_by_op_type(op_type)
        for node in nodes:
            bert_inputs = [node.input[i] for i in input_indices if i < len(node.input)]
            for bert_input in bert_inputs:
                if self.find_graph_input(bert_input):
                    if not casted:
                        graph_inputs.append(bert_input)
                elif bert_input in output_name_to_node:
                    parent = output_name_to_node[bert_input]
                    if parent.op_type == 'Cast' and self.find_graph_input(parent.input[0]) is not None:
                        if casted:
                            graph_inputs.append(parent.input[0])
        return graph_inputs

    def get_graph_inputs_from_fused_nodes(self, casted: bool):
        inputs = self.get_graph_inputs_from_node_type('EmbedLayerNormalization', [0, 1, 7], casted)
        inputs += self.get_graph_inputs_from_node_type('Attention', [3], casted)
        return inputs

    def change_input_to_int32(self):
        original_opset_version = self.model.opset_import[0].version
        graph = self.graph()

        new_graph_inputs = []
        casted_bert_graph_inputs = self.get_graph_inputs_from_fused_nodes(casted=True)

        for input in graph.input:
            if input.name in casted_bert_graph_inputs:
                self.utils.remove_cast_int32(input.name)
                int32_input = helper.make_tensor_value_info(input.name, TensorProto.INT32,
                                                            self.tensor_shape_to_list(input.type.tensor_type))
                new_graph_inputs.append(int32_input)
            else:
                new_graph_inputs.append(input)

        graph_def = helper.make_graph(graph.node,
                                      'int32 inputs',
                                      new_graph_inputs,
                                      graph.output,
                                      initializer=graph.initializer,
                                      value_info=graph.value_info)

        self.model = helper.make_model(graph_def, producer_name='onnxruntime-tools')

        # restore opset version
        self.model.opset_import[0].version = original_opset_version

    def use_dynamic_axes(self, dynamic_batch_dim='batch_size', dynamic_seq_len='max_seq_len'):
        """
        Update input and output shape to use dynamic axes.
        """
        bert_graph_inputs = self.get_graph_inputs_from_fused_nodes(
            casted=True) + self.get_graph_inputs_from_fused_nodes(casted=False)

        dynamic_batch_inputs = {}
        for input in self.model.graph.input:
            if input.name in bert_graph_inputs:
                dim_proto = input.type.tensor_type.shape.dim[0]
                dim_proto.dim_param = dynamic_batch_dim
                if dynamic_seq_len is not None:
                    dim_proto = input.type.tensor_type.shape.dim[1]
                    dim_proto.dim_param = dynamic_seq_len

        for output in self.model.graph.output:
            dim_proto = output.type.tensor_type.shape.dim[0]
            dim_proto.dim_param = dynamic_batch_dim

    def preprocess(self):
        self.adjust_reshape_and_expand()
        return

    def adjust_reshape_and_expand(self):
        nodes_to_remove = []
        for node in self.nodes():
            if node.op_type == 'Reshape':
                # Clean up unneccessary reshape nodes.
                # Find reshape nodes with no actually data in "shape" attribute and remove.
                reshape_shape = self.get_constant_value(node.input[1])
                if reshape_shape is not None and reshape_shape.size == 0:
                    nodes_to_remove.extend([node])
                    self.replace_input_of_all_nodes(node.output[0], node.input[0])
                    continue

                # Find path "Slice" -> "Reshape" -> "Expand" -> "Expand" -> current "Reshape", simplify the graph by
                # changing current reshape's input to output of slice.
                reshape_path = self.match_parent_path(node, ['Expand', 'Expand', 'Reshape', 'Slice'], [0, 0, 0, 0],
                                                      self.output_name_to_node())
                if reshape_path is not None:
                    expand_node = reshape_path[-3]
                    expand_shape_value = self.get_constant_value(expand_node.input[1])

                    reshape_before_expand = reshape_path[-2]
                    shape_value = self.get_constant_value(reshape_before_expand.input[1])

                    slice_node = reshape_path[-1]
                    if expand_shape_value is not None and shape_value is not None and len(
                            expand_shape_value) is 2 and len(
                                shape_value) is 1 and expand_shape_value[1] == shape_value[0]:
                        node.input[0] = slice_node.output[0]

        if nodes_to_remove:
            self.remove_nodes(nodes_to_remove)
            logger.info(f"Removed Reshape and Expand count: {len(nodes_to_remove)}")

    def clean_graph(self):
        output_name_to_node = self.output_name_to_node()
        nodes_to_remove = []
        for node in self.nodes():
            # Before:
            #  input_ids --> Shape --> Gather(indices=0) --> Unsqueeze ------+
            #          |                                                     |
            #          |                                                     v
            #          +----> Shape --> Gather(indices=1) --> Unsqueeze--->  Concat --> ConstantOfShape -->Cast --> EmbedLayerNormaliation/ReduceSum
            # After:
            #  input_ids --> Shape                                                  --> ConstantOfShape -->Cast --> EmbedLayerNormaliation/ReduceSum
            # TODO: merge ConstantOfShape -->Cast to ConstantOfShape (need update the data type of value)
            op_input_id = {"EmbedLayerNormalization": 1, "ReduceSum": 0, "Attention": 3}
            if node.op_type in op_input_id:
                i = op_input_id[node.op_type]
                parent_nodes = self.match_parent_path(
                    node, ['Cast', 'ConstantOfShape', 'Concat', 'Unsqueeze', 'Gather', 'Shape'], [i, 0, 0, 0, 0, 0],
                    output_name_to_node)
                if parent_nodes is not None:
                    cast, constantOfShape, concat, unsqueeze, gather, shape = parent_nodes
                    if shape.input[0] == self.graph().input[0].name:
                        constantOfShape.input[0] = shape.output[0]
                        output_name_to_node = self.output_name_to_node()

            if node.op_type == 'Attention':
                # Before:
                #   input_ids --> Shape -->ConstantOfShape -->Cast --> ReduceSum --> Attention
                # After:
                #   remove this path, and remove the optional mask_index input of Attention node.
                parent_nodes = self.match_parent_path(node, ['ReduceSum', 'Cast', 'ConstantOfShape', 'Shape'],
                                                      [3, 0, 0, 0], output_name_to_node)
                if parent_nodes is not None:
                    if parent_nodes[-1].input[0] == self.graph().input[0].name:
                        attention_node = helper.make_node('Attention',
                                                          inputs=node.input[0:len(node.input) - 1],
                                                          outputs=node.output,
                                                          name=node.name + "_remove_mask")
                        attention_node.domain = "com.microsoft"
                        attention_node.attribute.extend([helper.make_attribute("num_heads", self.num_heads)])
                        self.add_node(attention_node, self.get_graph_by_node(attention_node).name)
                        nodes_to_remove.append(node)
        self.remove_nodes(nodes_to_remove)

    def postprocess(self):
        self.clean_graph()
        self.prune_graph()

    def optimize(self, options: FusionOptions = None, add_dynamic_axes=False):
        if (options is None) or options.enable_layer_norm:
            self.fuse_layer_norm()

        if (options is None) or options.enable_gelu:
            self.fuse_gelu()

        self.preprocess()

        self.fuse_reshape()

        if (options is None) or options.enable_skip_layer_norm:
            self.fuse_skip_layer_norm()

        if (options is None) or options.enable_attention:
            if options is not None:
                self.attention_mask.set_mask_format(options.attention_mask_format)
            self.fuse_attention()

        if (options is None) or options.enable_embed_layer_norm:
            self.fuse_embed_layer()

        # Remove reshape nodes that having same shape of input and output based on symbolic shape inference.
        FusionUtils.remove_useless_reshape_nodes(self)

        self.postprocess()

        # Bias fusion is done after postprocess to avoid extra Reshape between bias and Gelu/FastGelu/SkipLayerNormalization
        if (options is None) or options.enable_bias_gelu:
            # Fuse Gelu and Add Bias before it.
            self.fuse_bias_gelu(is_fastgelu=True)
            self.fuse_bias_gelu(is_fastgelu=False)

        if (options is None) or options.enable_bias_skip_layer_norm:
            # Fuse SkipLayerNormalization and Add Bias before it.
            self.fuse_add_bias_skip_layer_norm()

        if (options is not None and options.enable_gelu_approximation):
            self.gelu_approximation()

        self.remove_unused_constant()

        # Use symbolic batch dimension in input and output.
        if add_dynamic_axes:
            self.use_dynamic_axes()

        logger.info(f"opset verion: {self.model.opset_import[0].version}")

    def get_fused_operator_statistics(self):
        """
        Returns node count of fused operators.
        """
        op_count = {}
        ops = [
            'EmbedLayerNormalization', 'Attention', 'Gelu', 'FastGelu', 'BiasGelu', 'LayerNormalization',
            'SkipLayerNormalization'
        ]
        for op in ops:
            nodes = self.get_nodes_by_op_type(op)
            op_count[op] = len(nodes)
        logger.info(f"Optimized operators:{op_count}")
        return op_count

    def is_fully_optimized(self):
        """
        Returns True when the model is fully optimized.
        """
        op_count = self.get_fused_operator_statistics()
        embed = op_count['EmbedLayerNormalization']
        attention = op_count['Attention']
        gelu = op_count['Gelu'] + op_count['BiasGelu'] + op_count['FastGelu']
        layer_norm = op_count['LayerNormalization'] + op_count['SkipLayerNormalization']
        is_perfect = (embed > 0) and (attention > 0) and (attention == gelu) and (layer_norm >= 2 * attention)

        if layer_norm == 0:
            logger.debug("Layer Normalization not fused")

        if gelu == 0:
            logger.debug("Gelu/FastGelu not fused")

        if embed == 0:
            logger.debug("Embed Layer not fused")

        if attention == 0:
            logger.warning("Attention not fused")

        return is_perfect
