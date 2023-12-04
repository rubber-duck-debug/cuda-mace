import torch
import math
from time import time
from mace_ops import cuda
from mace_ops.ops.invariant_message_passing import InvariantMessagePassingTP
from torch.autograd import Function

class InvariantMPTP(Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(node_attr, edge_attr, tp_weights, sender_list, receiver_list, first_occurences):
        output = torch.zeros(node_attr.size(0), edge_attr.shape[1], node_attr.shape[1], device=node_attr.device, dtype=node_attr.dtype)

        node_attr_sender = node_attr[sender_list]
        
        for i in range (edge_attr.shape[1]):
            
            out = node_attr_sender * edge_attr[:, i][:, None] * tp_weights[:, int(math.sqrt(i)), :]
            
            output[:, i, :].index_add_(0, receiver_list, out)

        return output
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        node_attr, edge_attr, tp_weights, sender_list, receiver_list, first_occurences = inputs
        ctx.save_for_backward(node_attr, edge_attr, tp_weights, sender_list, receiver_list, first_occurences)
        
    @staticmethod
    def backward(ctx, grad_output):
        node_attr, edge_attr, tp_weights, sender_list, receiver_list, first_occurences = ctx.saved_tensors
        
        gradY = torch.zeros_like(edge_attr)
        gradX = torch.zeros_like(node_attr)
        gradRadial = torch.zeros_like(tp_weights)
        
        for node in range (node_attr.shape[0]):
            
            edge_start = first_occurences[node]
            node_index = receiver_list[edge_start]
            edge_end = edge_attr.shape[0] if (node == first_occurences.shape[0] -1) else first_occurences[node + 1]
            
            gin = grad_output[node_index] # 16,128
            
            for edge in range (edge_start, edge_end):
                
                x = node_attr[sender_list[edge]]
                
                for m in range (16):
                    
                    L = int(math.sqrt(m))
                    
                    ylm = edge_attr[edge][m] # scalar
                    rad = tp_weights[edge][L] # 128
                    
                    gradRadial[edge, L, :] += ylm * gin[m] * x
                    gradY[edge, m] += (gin[m] * x * rad).sum()
                    
                    
        # for all edge indexes in sender_list[sort_sender_idx] need to update grads in [sender_list[sort_sender_idx]]
        sort_sender_idx = torch.argsort(sender_list).int()
        first_occurences = torch.ops.invariant_tp.calculate_first_occurences(sender_list, node_attr.shape[0], 64, sort_sender_idx)
       
        for node in range (node_attr.shape[0]):
            edge_start = first_occurences[node]
            node_index = sender_list[sort_sender_idx[edge_start]]
            edge_end = edge_attr.shape[0] if (node == first_occurences.shape[0] -1) else first_occurences[node + 1]
            gin = grad_output[node_index]
            
            for edge in range (edge_start, edge_end):
                
                for m in range (16):
        
                    L = int(math.sqrt(m))

                    ylm = edge_attr[sort_sender_idx[edge]][m] # scalar
                    rad = tp_weights[sort_sender_idx[edge]][L] # 128
                    
                    gradX[node_index] += gin[m] * rad * ylm
        
        return gradX, gradY, gradRadial, None, None, None
        

def reference(X, Y,  radial, receiver_list, nnodes ):

    output = torch.zeros(nnodes, Y.shape[1], X.shape[1], device=X.device, dtype=X.dtype)

    for i in range (Y.shape[1]):
        
        #print (X.shape, Y[:, i].shape)
        out = X * Y[:, i][:, None] * radial[:, int(math.sqrt(i)), :]
        
        output[:, i, :].index_add_(0, receiver_list, out)

    return output

def check_output(output_ref, output_cuda, name='output'):
    idx = torch.where(torch.abs(output_ref - output_cuda) >0)
    if (len (idx[0]) > 0):
        print (f"possible issue with {name}...")
        print (f"ndiffs: {len(idx)}")
        print ("max diff:", torch.max(torch.abs(output_ref[idx] - output_cuda[idx])), "rel. to largest:", torch.max(torch.abs(output_ref[idx] - output_cuda[idx]))/torch.max(torch.abs(output_ref[idx])))
        
def check_correctness(node_feats, edge_attrs, tp_weights, sender_list, receiver_list, nnodes):
    
    print ("sender list:", sender_list, sender_list.dtype, sender_list.shape)
    print ("receiver list:", receiver_list, receiver_list.dtype ,receiver_list.shape)
    
    sort_sender_idx = torch.argsort(sender_list)
    
    print (sender_list[sort_sender_idx], receiver_list[sort_sender_idx])
    
    tp = InvariantMessagePassingTP()
    first_occurences = tp.calculate_first_occurences(receiver_list, nnodes, torch.Tensor().int())
    
    print (first_occurences)
    print (first_occurences.dtype ,first_occurences.shape)
    
    node_feats_cuda = node_feats.clone().detach().requires_grad_(True)
    edge_attrs_cuda = edge_attrs.clone().detach().requires_grad_(True)
    tp_weights_cuda = tp_weights.clone().detach().requires_grad_(True)
    
    node_feats_python = node_feats.clone().detach().requires_grad_(True)
    edge_attrs_python = edge_attrs.clone().detach().requires_grad_(True)
    tp_weights_python = tp_weights.clone().detach().requires_grad_(True)
    
    node_feats_ref = node_feats.clone().detach().requires_grad_(True)
    node_feats_ref_sampled =  node_feats_ref[sender_list.int()]
    edge_attrs_ref = edge_attrs.clone().detach().requires_grad_(True)
    tp_weights_ref = tp_weights.clone().detach().requires_grad_(True)
    
    node_feats_e3nn = node_feats.clone().detach().requires_grad_(True)
    node_feats_e3nn_sampled =  node_feats_e3nn[sender_list.int()]
    edge_attrs_e3nn = edge_attrs.clone().detach().requires_grad_(True)
    
    tp_weights_e3nn = tp_weights.clone().detach().requires_grad_(True)
    tp_weights_rshaped_e3nn = tp_weights_e3nn.view(edge_attrs_cuda.shape[0], 4 * node_feats.shape[-1])
    
    #run the reference
    torch.cuda.synchronize()
    out_ref  = reference(node_feats_ref_sampled, edge_attrs_ref, tp_weights_ref, receiver_list, nnodes)
    torch.cuda.synchronize()
    t = out_ref.sum() * 2.0
    t.backward()
    torch.cuda.synchronize()
      
    out = InvariantMPTP.apply(
        node_feats_python,
        edge_attrs_python,
        tp_weights_python,
        sender_list,
        receiver_list, 
        first_occurences)
    torch.cuda.synchronize()
    osum = out.sum() * 2.0
    osum.backward()
    torch.cuda.synchronize()
    
    
    print ("Checking Python output vs ref backwards.")
    check_output(out_ref, out, "output")
    check_output(edge_attrs_ref.grad, edge_attrs_python.grad, "edge_attr grad")
    check_output(tp_weights_ref.grad, tp_weights_python.grad, "tp_weights grad")
    check_output(node_feats_ref.grad, node_feats_python.grad, "node feats grad")
    
    from e3nn import o3
    from mace.modules.irreps_tools import tp_out_irreps_with_instructions

    node_feats_irreps = o3.Irreps("96x0e")
    edge_attrs_irreps = o3.Irreps("1x0e+1x1o+1x2e+1x3o")
    target_irreps =o3.Irreps("96x0e+96x1o+96x2e+96x3o")
    
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
    
    osum = message.sum() * 2.0
    osum.backward()
    
    torch.cuda.synchronize()
    
    print ("Checking ref output vs e3nn backwards.")
    check_output(edge_attrs_ref.grad, edge_attrs_e3nn.grad, "E3NN edge_attr grad")
    check_output(tp_weights_ref.grad, tp_weights_e3nn.grad, "E3NN tp_weights grad")
    check_output(node_feats_ref.grad, node_feats_e3nn.grad, "E3NN node feats grad")
    
    tp = InvariantMessagePassingTP()
    first_occurences = tp.calculate_first_occurences(receiver_list, nnodes, torch.Tensor().int())
    
    cuda_out = tp.forward(node_feats_cuda, edge_attrs_cuda, tp_weights_cuda, sender_list, receiver_list, first_occurences)
    torch.cuda.synchronize()
    (cuda_out.sum() * 2.0).backward()
    torch.cuda.synchronize()
    
    print ("Checking ref output vs CUDA backwards.")
    check_output(edge_attrs_ref.grad, edge_attrs_cuda.grad, "CUDA edge_attr grad")
    check_output(tp_weights_ref.grad, tp_weights_cuda.grad, "CUDA tp_weights grad")
    check_output(node_feats_ref.grad, node_feats_cuda.grad, "CUDA node feats grad")


def accuracy(dtype, device):

    nedges = 300
    nnodes = 10
    nfeatures = 96
    L_MAX = 3
    nl = (L_MAX +1) ** 2

    print(f"--DTYPE: {dtype}")
    print(f"Benchmarking dtype {dtype} and device {device}")
    print(f"nodes: {nnodes} and edges: {nedges}")
    print(f"nfeatures: {nfeatures} and nsphericalharmonics: {nl}")

    node_feats = torch.randn((nnodes, nfeatures), dtype=dtype,
                   device=device, requires_grad=True)
    
    edge_attrs = torch.randn((nnodes, nl), dtype=dtype,
                   device=device, requires_grad=True)
    tp_weights = torch.randn((nnodes, L_MAX+1, nfeatures), dtype=dtype,
                   device=device, requires_grad=True) 
    
    receiver_list  = torch.sort(torch.randint(nnodes, (nedges,), device=device, dtype=torch.int32))[0]
    
    r=torch.randperm(receiver_list.shape[0])
    sender_list = receiver_list[r] #mimic sender_list by permutation
    
    edge_attrs = edge_attrs[receiver_list] - (edge_attrs[sender_list] + 0.5) #mimic pair list
    tp_weights = tp_weights[receiver_list] - (tp_weights[sender_list] + 0.5) #mimic pair list    
   
    check_correctness(node_feats, edge_attrs, tp_weights, sender_list, receiver_list, nnodes)
    
def benchmark(dtype, device):

    nedges = 30000 * 5
    nnodes = 1000 * 5
    nfeatures = 96
    L_MAX = 3
    nl = (L_MAX +1) ** 2

    print(f"--DTYPE: {dtype}")
    print(f"Benchmarking dtype {dtype} and device {device}")
    print(f"nodes: {nnodes} and edges: {nedges}")
    print(f"nfeatures: {nfeatures} and nsphericalharmonics: {nl}")

    node_feats = torch.randn((nnodes, nfeatures), dtype=dtype,
                   device=device, requires_grad=True)
    
    edge_attrs = torch.randn((nnodes, nl), dtype=dtype,
                   device=device, requires_grad=True)
    tp_weights = torch.randn((nnodes, L_MAX+1, nfeatures), dtype=dtype,
                   device=device, requires_grad=True) 
    
    receiver_list  = torch.sort(torch.randint(nnodes, (nedges,), device=device, dtype=torch.int32))[0]
    
    r=torch.randperm(receiver_list.shape[0])
    sender_list = receiver_list[r] #mimic sender_list by permutation
    
    edge_attrs = edge_attrs[receiver_list] - (edge_attrs[sender_list] + 0.5) #mimic pair list
    tp_weights = tp_weights[receiver_list] - (tp_weights[sender_list] + 0.5) #mimic pair list

    tp = InvariantMessagePassingTP()
    
    node_feats_cuda = node_feats.clone().detach().requires_grad_(True)
    edge_attrs_cuda = edge_attrs.clone().detach().requires_grad_(True)
    tp_weights_cuda = tp_weights.clone().detach().requires_grad_(True)
    
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for i in range (1000):
        first_occurences = tp.calculate_first_occurences(receiver_list, nnodes, torch.Tensor().int())
        cuda_out = tp.forward(node_feats_cuda, edge_attrs_cuda, tp_weights_cuda, sender_list, receiver_list, first_occurences)
        os = cuda_out.sum() * 2.0
        os.backward()
        
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    
if __name__ == "__main__":
    accuracy(torch.float32, "cuda")
    #benchmark(torch.float32, "cuda") # run this with nsys nvprof python3 examples/InvariantMessagePassing.py
