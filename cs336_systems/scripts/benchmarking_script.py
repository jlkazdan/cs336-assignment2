import torch
import timeit
from timeit import default_timer as timer
import argparse
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy


def forward_test(model, args):
    tokens = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length)).unsqueeze(1)
    logits = model(tokens)

def backward_test(model, optimizer, args):
    tokens = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length))
    targets = tokens[..., 1:]
    new_target = torch.randint(1, args.vocab_size, (args.batch_size,1))
    targets = torch.cat([targets, new_target], -1)
    begin_forward = timer()

    logits = model(tokens)
    end_forward = timer()
    loss = cross_entropy(logits, targets)
    
    begin_backward = timer()
    loss.backward()
    end_backward = timer()

    optimizer.step()

    forward_time = end_forward - begin_forward
    backward_time = end_backward - begin_backward
    return forward_time, backward_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default = 'cpu')
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
    args = parser.parse_args()

    transformer_model = BasicsTransformerLM(vocab_size = args.vocab_size,
                context_length= args.context_length,
                d_model = args.d_model,
                num_layers = args.num_layers,
                num_heads = args.num_heads,
                d_ff = args.d_ff,
                rope_theta = args.rope_theta,
                )

    optimizer = AdamW(transformer_model.parameters())

    for i in range(args.warmup_steps):
        backward_test(transformer_model, optimizer, args)
        #torch.cuda.synchronize()
    forward, backward = [], []
    for i in range(args.steps_timed):
        f, b = backward_test(transformer_model, optimizer, args)
        forward.append(f)
        backward.append(b)
        #torch.cuda.synchronize()
    print(forward)
    print(backward)

    #time the forward function

if __name__ == '__main__':
    main()