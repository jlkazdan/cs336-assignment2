import numpy as np
import torch
import timeit
import argparse
from timeit import default_timer as timer
from torch.profiler import profile
import torch.cuda.nvtx as nvtx
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy



def forward_test(model, args):
    tokens = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length)).unsqueeze(1)
    logits = model(tokens)

def backward_test(model, optimizer, args):
    tokens = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length))
    targets = tokens[..., 1:].to(args.device)
    new_target = torch.randint(1, args.vocab_size, (args.batch_size,1)).to(args.device)
    targets = torch.cat([targets, new_target], -1).to(args.device)
    begin_forward = timer()

    with nvtx.range('forward pass'):
        logits = model(tokens)
    end_forward = timer()
    loss = cross_entropy(logits, targets)
    
    begin_backward = timer()
    with nvtx.range('backward pass'):
        loss.backward()
    end_backward = timer()

    optimizer.step()

    forward_time = end_forward - begin_forward
    backward_time = end_backward - begin_backward
    return forward_time, backward_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default = 'cuda')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--vocab_size', type = int, default = 10000)
    parser.add_argument('--num_layers', type = int, default = 4)
    parser.add_argument('--context_length', type = int, default = 512)
    parser.add_argument('--d_model', type = int, default = 512)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=1344)
    parser.add_argument('--remove_rmsnorm', type=bool, default=False)
    parser.add_argument('--use_post_norm', type=bool, default=False)
    parser.add_argument('--remove_rope', type=bool, default=False)
    parser.add_argument('--rope_theta', type=int, default=10000)
    parser.add_argument('--ffn_type', type=str, default = None)
    parser.add_argument('--steps_timed', type = int, default = 10)
    parser.add_argument('--warmup_steps', type=int, default=5)
    parser.add_argument('--n_steps', type=int, default=3)
    args = parser.parse_args()
    device = args.device
    transformer_model = BasicsTransformerLM(vocab_size = args.vocab_size,
                context_length= args.context_length,
                d_model = args.d_model,
                num_layers = args.num_layers,
                num_heads = args.num_heads,
                d_ff = args.d_ff,
                rope_theta = args.rope_theta,
                ).to(device)

    optimizer = AdamW(transformer_model.parameters())
    with profile(
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ], 
        schedule = torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=args.n_steps),
        experimental_config = torch._C._profiler._ExperimentalConfig(verbose=True),
        record_shapes = True,
        profile_memory = True,
        with_stack=True,
    ) as prof:
    
        for i in range(args.warmup_steps):
            with nvtx.range(f'warmup pass {i}'):
                backward_test(transformer_model, optimizer, args)
                torch.cuda.synchronize()
                prof.step()
        forward, backward = [], []
        for i in range(args.steps_timed):
            with nvtx.range(f'pass {i}'):
                f, b = backward_test(transformer_model, optimizer, args)
                forward.append(f)
                backward.append(b)
                torch.cuda.synchronize()
                prof.step()
        prof.export_memory_timeline('timeline.html', device=device)
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

        forward =  np.array(forward)
        backward = np.array(backward)
        print(f'The mean time for the forward pass is {forward.mean()}')
        print(f'The std of the forward pass is {forward.std()}')
        print(f'The mean time for the backward pass is {backward.mean()}')
        print(f'The std of the backward pass is {backward.std()}')

    #time the forward function

if __name__ == '__main__':
    main()