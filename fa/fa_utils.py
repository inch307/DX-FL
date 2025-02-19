import torch
from fa.fa_conv import FeedbackConvLayer
from fa.fa_linear import FeedbackLinearLayer


def sync_B(global_model, args, round_counter):
    print(f'round_counter: {round_counter}, args: {args.sync_round}')
    if round_counter == args.sync_round:
        with torch.no_grad():
            for n, param in global_model.named_modules():
                if isinstance(param, FeedbackConvLayer) or isinstance(param, FeedbackLinearLayer):
                    param.B.copy_(param.weight.data.clone())
        round_counter = 0
    return round_counter