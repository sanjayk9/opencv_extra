# Small attention blocks in the forms that common exporters write, for the attention fusion tests.
# Each keeps the node pattern of its source exactly, with a small size: batch 1, 10 tokens, 4 heads of 8.
#
#   attention_vit_runtime_scale  timm ViT/DeiT: one QKV projection, Reshape [B,S,3,H,D], Transpose [2,0,3,1,4],
#                                Split + Squeeze, and the scale computed at run time from Q's shape as
#                                sqrt(1/sqrt(D)) on both Q and K^T
#   attention_beit_rel_bias      timm BEiT: as above with -1 for D in the Reshape and a relative position bias
#                                added to the scores before Softmax
#   attention_siglip_mul_scale   HF SigLIP: separate Q/K/V projections with shapes built from Shape/Gather/Concat,
#                                K transposed with perm [0,2,3,1], a scalar Mul after Q*K^T and two float Casts
#                                after Softmax
#   attention_hiera_unbind       SAM2 Hiera: QKV split on axis 2 with no Transpose before it, then one Transpose
#                                per part and a constant scale on Q and K
#
# Writes models/<name>.onnx, data/input_<name>.npy and data/output_<name>.npy (outputs from ONNX Runtime).
# Run from testdata/dnn/onnx.

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper
import onnxruntime as ort

B, S, H, D = 1, 10, 4, 8
E = H * D

rng = np.random.default_rng(0)


def const(name, array):
    array = np.asarray(array)
    return helper.make_node("Constant", [], [name], name=name, value=numpy_helper.from_array(array, name + "_value"))


def ints(name, values):
    return const(name, np.array(values, dtype=np.int64))


def floats(name, values):
    return const(name, np.array(values, dtype=np.float32))


def weight(name, shape, scale):
    return floats(name, rng.standard_normal(shape).astype(np.float32) * scale)


def save(name, nodes, out_shape=(B, S, E)):
    graph = helper.make_graph(nodes, name,
                              [helper.make_tensor_value_info("input", TensorProto.FLOAT, [B, S, E])],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT, list(out_shape))])
    model = helper.make_model(graph, producer_name="opencv-attention-fusion-tests",
                              opset_imports=[helper.make_operatorsetid("", 17)])
    model.ir_version = 8
    checker.check_model(model)
    onnx.save(model, "models/{}.onnx".format(name))

    x = rng.standard_normal((B, S, E)).astype(np.float32)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    y = ort.InferenceSession("models/{}.onnx".format(name), so, providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    np.save("data/input_{}.npy".format(name), x)
    np.save("data/output_{}.npy".format(name), y)
    print("saved", name, y.shape)


def runtime_scale(prefix, src):
    """sqrt(1 / sqrt(last dim of src)), as timm exports the scale applied to both Q and K^T."""
    return [
        helper.make_node("Shape", [src], [prefix + "shape"]),
        ints(prefix + "start", [-1]),
        ints(prefix + "end", [9223372036854775807]),
        helper.make_node("Slice", [prefix + "shape", prefix + "start", prefix + "end"], [prefix + "dim"]),
        helper.make_node("Cast", [prefix + "dim"], [prefix + "dim_f"], to=TensorProto.FLOAT),
        helper.make_node("Sqrt", [prefix + "dim_f"], [prefix + "sqrt_d"]),
        floats(prefix + "one", [1.0]),
        helper.make_node("Div", [prefix + "one", prefix + "sqrt_d"], [prefix + "inv_sqrt_d"]),
    ]


def timm_block(name, reshape_d, rel_bias):
    nodes = [
        weight("qkv_w", (E, 3 * E), 0.2),
        helper.make_node("MatMul", ["input", "qkv_w"], ["qkv_mm"]),
        weight("qkv_b", (3 * E,), 0.1),
        helper.make_node("Add", ["qkv_b", "qkv_mm"], ["qkv"]),
        ints("qkv_shape", [B, S, 3, H, reshape_d]),
        helper.make_node("Reshape", ["qkv", "qkv_shape"], ["qkv_5d"]),
        helper.make_node("Transpose", ["qkv_5d"], ["qkv_t"], perm=[2, 0, 3, 1, 4]),
        ints("split_sizes", [1, 1, 1]),
        helper.make_node("Split", ["qkv_t", "split_sizes"], ["q_s", "k_s", "v_s"], axis=0),
        ints("q_axes", [0]),
        helper.make_node("Squeeze", ["q_s", "q_axes"], ["q"]),
    ]
    nodes += runtime_scale("scale_", "q")
    nodes += [
        helper.make_node("Sqrt", ["scale_inv_sqrt_d"], ["q_scale"]),
        helper.make_node("Mul", ["q", "q_scale"], ["q_scaled"]),
        ints("k_axes", [0]),
        helper.make_node("Squeeze", ["k_s", "k_axes"], ["k"]),
        helper.make_node("Transpose", ["k"], ["k_t"], perm=[0, 1, 3, 2]),
        helper.make_node("Sqrt", ["scale_inv_sqrt_d"], ["k_scale"]),
        helper.make_node("Mul", ["k_t", "k_scale"], ["k_scaled"]),
        helper.make_node("MatMul", ["q_scaled", "k_scaled"], ["scores"]),
    ]
    scores = "scores"
    if rel_bias:
        nodes += [weight("rel_bias", (1, H, S, S), 1.0),
                  helper.make_node("Add", ["scores", "rel_bias"], ["scores_biased"])]
        scores = "scores_biased"
    nodes += [
        helper.make_node("Softmax", [scores], ["probs"], axis=-1),
        ints("v_axes", [0]),
        helper.make_node("Squeeze", ["v_s", "v_axes"], ["v"]),
        helper.make_node("MatMul", ["probs", "v"], ["ctx"]),
        helper.make_node("Transpose", ["ctx"], ["ctx_t"], perm=[0, 2, 1, 3]),
        ints("out_shape", [B, S, E]),
        helper.make_node("Reshape", ["ctx_t", "out_shape"], ["output"]),
    ]
    save(name, nodes)


def siglip_heads_shape(prefix, src):
    """[batch, tokens, H, D] built at run time from src's shape, as HF exports it."""
    return [
        helper.make_node("Shape", [src], [prefix + "shape"]),
        ints(prefix + "i0", 0),
        helper.make_node("Gather", [prefix + "shape", prefix + "i0"], [prefix + "b"], axis=0),
        ints(prefix + "i1", 1),
        helper.make_node("Gather", [prefix + "shape", prefix + "i1"], [prefix + "s"], axis=0),
        ints(prefix + "ax", [0]),
        helper.make_node("Unsqueeze", [prefix + "b", prefix + "ax"], [prefix + "b1"]),
        helper.make_node("Unsqueeze", [prefix + "s", prefix + "ax"], [prefix + "s1"]),
        ints(prefix + "h", [H]),
        ints(prefix + "d", [D]),
        helper.make_node("Concat", [prefix + "b1", prefix + "s1", prefix + "h", prefix + "d"], [prefix + "dims"], axis=0),
    ]


def siglip_block(name):
    nodes = []
    for p in ("q", "k", "v"):
        nodes += [
            weight(p + "_w", (E, E), 0.2),
            helper.make_node("MatMul", ["input", p + "_w"], [p + "_mm"]),
            weight(p + "_b", (E,), 0.1),
            helper.make_node("Add", [p + "_b", p + "_mm"], [p + "_proj"]),
        ]
        nodes += siglip_heads_shape(p + "_shape_", "input")
        nodes += [helper.make_node("Reshape", [p + "_proj", p + "_shape_dims"], [p + "_4d"]),
                  helper.make_node("Transpose", [p + "_4d"], [p + "_t"],
                                   perm=[0, 2, 3, 1] if p == "k" else [0, 2, 1, 3])]
    nodes += [
        helper.make_node("MatMul", ["q_t", "k_t"], ["scores"]),
        floats("scale", np.float32(1.0 / np.sqrt(D))),
        helper.make_node("Mul", ["scores", "scale"], ["scores_scaled"]),
        helper.make_node("Softmax", ["scores_scaled"], ["probs"], axis=-1),
        helper.make_node("Cast", ["probs"], ["probs_c1"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["probs_c1"], ["probs_c2"], to=TensorProto.FLOAT),
        helper.make_node("MatMul", ["probs_c2", "v_t"], ["ctx"]),
        helper.make_node("Transpose", ["ctx"], ["ctx_t"], perm=[0, 2, 1, 3]),
        ints("out_shape", [B, S, E]),
        helper.make_node("Reshape", ["ctx_t", "out_shape"], ["output"]),
    ]
    save(name, nodes)


def hiera_block(name):
    scale = np.float32(D ** -0.25)
    nodes = [
        weight("qkv_w", (E, 3 * E), 0.2),
        helper.make_node("MatMul", ["input", "qkv_w"], ["qkv_mm"]),
        weight("qkv_b", (3 * E,), 0.1),
        helper.make_node("Add", ["qkv_b", "qkv_mm"], ["qkv"]),
        ints("qkv_shape", [B, S, 3, H, D]),
        helper.make_node("Reshape", ["qkv", "qkv_shape"], ["qkv_5d"]),
        ints("split_sizes", [1, 1, 1]),
        helper.make_node("Split", ["qkv_5d", "split_sizes"], ["q_s", "k_s", "v_s"], axis=2),
    ]
    for p, perm in (("q", [0, 2, 1, 3]), ("k", [0, 2, 3, 1]), ("v", [0, 2, 1, 3])):
        nodes += [ints(p + "_axes", [2]),
                  helper.make_node("Squeeze", [p + "_s", p + "_axes"], [p]),
                  helper.make_node("Transpose", [p], [p + "_t"], perm=perm)]
    nodes += [
        floats("q_scale", [scale]),
        helper.make_node("Mul", ["q_t", "q_scale"], ["q_scaled"]),
        floats("k_scale", [scale]),
        helper.make_node("Mul", ["k_t", "k_scale"], ["k_scaled"]),
        helper.make_node("MatMul", ["q_scaled", "k_scaled"], ["scores"]),
        helper.make_node("Softmax", ["scores"], ["probs"], axis=-1),
        helper.make_node("MatMul", ["probs", "v_t"], ["ctx"]),
        helper.make_node("Transpose", ["ctx"], ["ctx_t"], perm=[0, 2, 1, 3]),
        ints("out_shape", [B, S, E]),
        helper.make_node("Reshape", ["ctx_t", "out_shape"], ["output"]),
    ]
    save(name, nodes)


timm_block("attention_vit_runtime_scale", D, rel_bias=False)
timm_block("attention_beit_rel_bias", -1, rel_bias=True)
siglip_block("attention_siglip_mul_scale")
hiera_block("attention_hiera_unbind")
