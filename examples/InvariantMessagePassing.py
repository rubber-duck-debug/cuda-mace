import torch
import math
from time import time

import mace_ops
from mace_ops.ops.invariant_message_passing import (
    InvariantMessagePassingTP,
)
from torch.autograd import Function
from torch.autograd import gradcheck


class InvariantMPTP(Function):
    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(node_attr, edge_attr, tp_weights, sender_list, receiver_list):
        output = torch.zeros(
            node_attr.size(0),
            edge_attr.shape[1],
            node_attr.shape[1],
            device=node_attr.device,
            dtype=node_attr.dtype,
        )

        node_attr_sender = node_attr[sender_list]

        for i in range(edge_attr.shape[1]):
            out = (
                node_attr_sender
                * edge_attr[:, i][:, None]
                * tp_weights[:, int(math.sqrt(i)), :]
            )

            output[:, i, :].index_add_(0, receiver_list, out)

        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        node_attr, edge_attr, tp_weights, sender_list, receiver_list = inputs

        first_occurences_fwd = torch.ops.invariant_tp.calculate_first_occurences(
            receiver_list, node_attr.shape[0], 64
        )

        ctx.save_for_backward(
            node_attr,
            edge_attr,
            tp_weights,
            sender_list,
            receiver_list,
            first_occurences_fwd,
        )

    @staticmethod
    def backward(ctx, grad_output):
        (
            node_attr,
            edge_attr,
            tp_weights,
            sender_list,
            receiver_list,
            first_occurences,
        ) = ctx.saved_tensors

        gradY = torch.zeros_like(
            edge_attr, device=node_attr.device, dtype=node_attr.dtype
        )
        gradX_edges = torch.zeros(
            edge_attr.shape[0],
            node_attr.shape[1],
            device=node_attr.device,
            dtype=node_attr.dtype,
        )
        gradX = torch.zeros_like(
            node_attr, device=node_attr.device, dtype=node_attr.dtype
        )
        gradRadial = torch.zeros_like(
            tp_weights, device=node_attr.device, dtype=node_attr.dtype
        )

        for node in range(node_attr.shape[0]):
            edge_start = first_occurences[node]
            node_index = receiver_list[edge_start]
            edge_end = (
                edge_attr.shape[0]
                if (node == first_occurences.shape[0] - 1)
                else first_occurences[node + 1]
            )

            gin = grad_output[node_index]  # 16,128

            for edge in range(edge_start, edge_end):
                x = node_attr[sender_list[edge]]

                for m in range(16):
                    L = int(math.sqrt(m))

                    ylm = edge_attr[edge][m]  # scalar
                    rad = tp_weights[edge][L]  # 128

                    gradRadial[edge, L, :] += ylm * gin[m] * x
                    gradY[edge, m] += (gin[m] * x * rad).sum()
                    gradX_edges[edge, :] += ylm * rad

        # for all edge indexes in sender_list[sort_sender_idx] need to update grads in [sender_list[sort_sender_idx]]
        sort_sender_idx = torch.argsort(sender_list).int()
        first_occurences = torch.ops.invariant_tp.calculate_first_occurences_with_sort(
            sender_list, node_attr.shape[0], 64, sort_sender_idx
        )

        # print (receiver_list[sort_sender_idx])

        #print(receiver_list)
        #print(sender_list)
        #print (receiver_list[sort_sender_idx])
        #print (sender_list[sort_sender_idx])
        #print (receiver_list[sort_sender_idx][torch.argsort(receiver_list[sort_sender_idx])])
        #print (sender_list[sort_sender_idx][torch.argsort(receiver_list[sort_sender_idx])])
        
        for node in range(node_attr.shape[0]):
            edge_start = first_occurences[node]
            node_index = sender_list[sort_sender_idx[edge_start]]
            edge_end = (
                edge_attr.shape[0]
                if (node == first_occurences.shape[0] - 1)
                else first_occurences[node + 1]
            )  

            for edge in range(edge_start, edge_end):
                
                gin = grad_output[receiver_list[sort_sender_idx[edge]]]

                tmp = 0.0
                for m in range(16):
                    L = int(math.sqrt(m))

                    ylm = edge_attr[sort_sender_idx[edge]][m]  # scalar
                    rad = tp_weights[sort_sender_idx[edge]][L]  # 128

                    gradX[node_index] += gin[m] * rad * ylm
                    # gradX[node_index] += gin[m] * gradX_edges[sort_sender_idx[edge]]

        return gradX, gradY, gradRadial, None, None, None


def reference(X, Y, radial, receiver_list, nnodes):
    output = torch.zeros(nnodes, Y.shape[1], X.shape[1], device=X.device, dtype=X.dtype)

    for i in range(Y.shape[1]):
        # print (X.shape, Y[:, i].shape)
        out = X * Y[:, i][:, None] * radial[:, int(math.sqrt(i)), :]

        output[:, i, :].index_add_(0, receiver_list, out)

    return output


def check_output(output_ref, output_cuda, name="output"):
    idx = torch.where(torch.abs(output_ref - output_cuda) > 0)
    if len(idx[0]) > 0:
        print(f"possible issue with {name}...")
        print(f"ndiffs: {len(idx)}")
        print(
            "max diff:",
            torch.max(torch.abs(output_ref[idx] - output_cuda[idx])),
            "rel. to largest:",
            torch.max(torch.abs(output_ref[idx] - output_cuda[idx]))
            / torch.max(torch.abs(output_ref[idx])),
        )


def check_correctness(
    node_feats, edge_attrs, tp_weights, sender_list, receiver_list, nnodes
):
    print(node_feats.dtype)
    nfeatures = node_feats.shape[-1]
    print("sender list:", sender_list, sender_list.dtype, sender_list.shape)
    print("receiver list:", receiver_list, receiver_list.dtype, receiver_list.shape)

    sort_sender_idx = torch.argsort(sender_list)

    node_feats_cuda = node_feats.clone().detach().requires_grad_(True)
    edge_attrs_cuda = edge_attrs.clone().detach().requires_grad_(True)
    tp_weights_cuda = tp_weights.clone().detach().requires_grad_(True)

    node_feats_python = node_feats.clone().detach().requires_grad_(True)
    edge_attrs_python = edge_attrs.clone().detach().requires_grad_(True)
    tp_weights_python = tp_weights.clone().detach().requires_grad_(True)

    node_feats_ref = node_feats.clone().detach().double().requires_grad_(True)
    edge_attrs_ref = edge_attrs.clone().detach().double().requires_grad_(True)
    tp_weights_ref = tp_weights.clone().detach().double().requires_grad_(True)

    node_feats_e3nn = node_feats.clone().detach().requires_grad_(True)
    node_feats_e3nn_sampled = node_feats_e3nn[sender_list.int()]
    edge_attrs_e3nn = edge_attrs.clone().detach().requires_grad_(True)
    tp_weights_e3nn = tp_weights.clone().detach().requires_grad_(True)
    tp_weights_rshaped_e3nn = tp_weights_e3nn.view(
        edge_attrs_cuda.shape[0], 4 * node_feats.shape[-1]
    )

    node_feats_gradcheck = node_feats.clone().detach().requires_grad_(True)
    edge_attrs_gradcheck = edge_attrs.clone().detach().requires_grad_(True)
    tp_weights_gradcheck = tp_weights.clone().detach().requires_grad_(True)

    # run the reference
    """
    torch.cuda.synchronize()
    out_ref = reference(
        node_feats_ref[sender_list],
        edge_attrs_ref,
        tp_weights_ref,
        receiver_list,
        nnodes,
    )
    torch.cuda.synchronize()
    t = (out_ref**2.0).sum()
    t.backward()
    torch.cuda.synchronize()

    out = InvariantMPTP.apply(
        node_feats_python,
        edge_attrs_python,
        tp_weights_python,
        sender_list,
        receiver_list,
    )
    torch.cuda.synchronize()
    osum = (out**2.0).sum()
    osum.backward()
    torch.cuda.synchronize()

    print("Checking Python output vs ref backwards.")
    check_output(out_ref, out, "output")
    check_output(edge_attrs_ref.grad, edge_attrs_python.grad, "edge_attr grad")
    check_output(tp_weights_ref.grad, tp_weights_python.grad, "tp_weights grad")
    check_output(node_feats_ref.grad, node_feats_python.grad, "node feats grad")
    """
    
    from e3nn import o3
    from mace.modules.irreps_tools import tp_out_irreps_with_instructions

    node_feats_irreps = o3.Irreps(f"{nfeatures}x0e")
    edge_attrs_irreps = o3.Irreps("1x0e+1x1o+1x2e+1x3o")
    target_irreps =o3.Irreps(f"{96}x0e+{nfeatures}x1o+{nfeatures}x2e+{nfeatures}x3o")
    
    irreps_mid, instructions = tp_out_irreps_with_instructions(
                node_feats_irreps, edge_attrs_irreps, target_irreps
            )

    conv_tp = o3.TensorProduct(
                node_feats_irreps,
                edge_attrs_irreps,
                irreps_mid,
                instructions=instructions,
                shared_weights=False,
                internal_weights=False,
            )

    mji = conv_tp(node_feats_e3nn_sampled, edge_attrs_e3nn, tp_weights_rshaped_e3nn)
    
    from mace.tools.scatter import scatter_sum
    
    message = scatter_sum(src=mji, index=receiver_list.long(), dim=0, dim_size=node_feats.shape[0])
    
    osum = (message** 2.0).sum() 
    osum.backward()
    
    torch.cuda.synchronize()
    
    #print ("Checking ref output vs e3nn backwards.")
    #check_output(edge_attrs_ref.grad, edge_attrs_e3nn.grad, "E3NN edge_attr grad")
    #check_output(tp_weights_ref.grad, tp_weights_e3nn.grad, "E3NN tp_weights grad")
    #check_output(node_feats_ref.grad, node_feats_e3nn.grad, "E3NN node feats grad")
    
    
    tp = InvariantMessagePassingTP()

    cuda_out = tp.forward(node_feats_cuda, edge_attrs_cuda, tp_weights_cuda, sender_list, receiver_list, nnodes)

    
    torch.cuda.synchronize()
    osum = (cuda_out ** 2.0).sum()
    osum.backward()
    torch.cuda.synchronize()

    print (node_feats_e3nn.grad - node_feats_cuda.grad)
    
    idx = torch.where(node_feats_e3nn.grad - node_feats_cuda.grad> 1e-5)
    
    
    print (idx[0])
    print (idx[1])
    print ((node_feats_e3nn.grad - node_feats_cuda.grad)[idx])

    
    print (len(idx[0]), node_feats_cuda.grad.numel())
    
    #print ("Checking e3nn output vs CUDA forwards.")
    #check_output(message, cuda_out, "CUDA forward")
    print ("Checking e3nn output vs CUDA backwards.")
    check_output(edge_attrs_e3nn.grad, edge_attrs_cuda.grad, "CUDA edge_attr grad")
    check_output(tp_weights_e3nn.grad, tp_weights_cuda.grad, "CUDA tp_weights grad")
    check_output(node_feats_e3nn.grad, node_feats_cuda.grad, "CUDA node feats grad")

    #print("Checking CUDA implementation via finite difference.")
    #print(gradcheck(tp.forward, (node_feats_gradcheck.double(), edge_attrs_gradcheck.double(), tp_weights_gradcheck.double(), sender_list, receiver_list, nnodes), eps=1e-5, atol=1e-5))


def accuracy(dtype, device):
    node_feats = torch.load("node_feats.pt")
    edge_attrs = torch.load("edge_attrs.pt")
    tp_weights = torch.load("tp_weights.pt")
    receiver_list = torch.load("receiver.pt")
    sender_list = torch.load("sender.pt")
    
    print(f"--DTYPE: {dtype}")
    print(f"Benchmarking dtype {dtype} and device {device}")
    print(f"nodes: {node_feats.shape[0]} and edges: {edge_attrs.shape[0]}")
    print(f"nfeatures: {node_feats.shape[1]} and nsphericalharmonics: {edge_attrs.shape[1]}")
    check_correctness(
        node_feats, edge_attrs, tp_weights, sender_list.int(), receiver_list.int(), node_feats.shape[0]
    )


def benchmark(dtype, device):
    nnodes = 5800
    nedges = nnodes * 45

    nfeatures = 96
    L_MAX = 3
    nl = (L_MAX + 1) ** 2

    print(f"--DTYPE: {dtype}")
    print(f"Benchmarking dtype {dtype} and device {device}")
    print(f"nodes: {nnodes} and edges: {nedges}")
    print(f"nfeatures: {nfeatures} and nsphericalharmonics: {nl}")

    node_feats = torch.randn(
        (nnodes, nfeatures), dtype=dtype, device=device, requires_grad=True
    )

    edge_attrs = torch.randn(
        (nnodes, nl), dtype=dtype, device=device, requires_grad=True
    )
    tp_weights = torch.randn(
        (nnodes, L_MAX + 1, nfeatures), dtype=dtype, device=device, requires_grad=True
    )

    receiver_list = torch.sort(
        torch.randint(nnodes, (nedges,), device=device, dtype=torch.int32)
    )[0]

    r = torch.randperm(receiver_list.shape[0])
    sender_list = receiver_list[r]  # mimic sender_list by permutation

    edge_attrs = edge_attrs[receiver_list] - (
        edge_attrs[sender_list] + 0.5
    )  # mimic pair list
    tp_weights = tp_weights[receiver_list] - (
        tp_weights[sender_list] + 0.5
    )  # mimic pair list

    tp = InvariantMessagePassingTP()

    node_feats_cuda_old = node_feats.clone().detach().requires_grad_(True)
    edge_attrs_cuda_old = edge_attrs.clone().detach().requires_grad_(True)
    tp_weights_cuda_old = tp_weights.clone().detach().requires_grad_(True)

    node_feats_cuda = node_feats.clone().detach().requires_grad_(True)
    edge_attrs_cuda = edge_attrs.clone().detach().requires_grad_(True)
    tp_weights_cuda = tp_weights.clone().detach().requires_grad_(True)

    torch.cuda.synchronize()
    start = time()
    torch.cuda.cudart().cudaProfilerStart()
    for i in range(1000):
        cuda_out = tp.forward(
            node_feats_cuda,
            edge_attrs_cuda,
            tp_weights_cuda,
            sender_list,
            receiver_list,
            nnodes,
        )
        os = cuda_out.sum() * 2.0
        os.backward()
    torch.cuda.synchronize()
    end = time()
    print(end - start)
    torch.cuda.cudart().cudaProfilerStop()

if __name__ == "__main__":
    #accuracy(torch.float32, "cuda")
    benchmark(torch.float32, "cuda") # run this with nsys nvprof python3 examples/InvariantMessagePassing.py
