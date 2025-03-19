import torch
import torch.nn as nn

from fa.fa_conv import FeedbackConvLayer
from fa.fa_linear import FeedbackLinearLayer


def sync_B(global_model, args, round_counter):
    if round_counter == args.sync_round:
        print(f'round_counter: {round_counter}, args: {args.sync_round}, sync')
        with torch.no_grad():
            for n, param in global_model.named_modules():
                if isinstance(param, FeedbackConvLayer) or isinstance(param, FeedbackLinearLayer):
                    param.B.copy_(param.weight.data.clone())
        round_counter = 0
    else:
        if args.sync_round > 1 and round_counter >= 1:
            print(f'round_counter: {round_counter}, args: {args.sync_round}, scale')
            for m in global_model.modules():
                if isinstance(m, FeedbackConvLayer) or isinstance(m, FeedbackLinearLayer):
                    with torch.no_grad():
                        m.scale_B()
    return round_counter

def convert_feedback_to_bp(module, device):
    for name, child in module.named_children():
        if isinstance(child, FeedbackConvLayer):
            conv_params = {
                "in_channels": child.in_channels,
                "out_channels": child.out_channels,
                "kernel_size": child.kernel_size,
                "stride": child.stride,
                "padding": child.padding,
                "dilation": child.dilation,
                "groups": child.groups,
                "bias": child.bias is not None,
                "padding_mode": child.padding_mode if hasattr(child, "padding_mode") else "zeros",
            }
            new_conv = nn.Conv2d(**conv_params)
            new_conv = new_conv.to(device)
            new_conv.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_conv.bias.data.copy_(child.bias.data)

            setattr(module, name, new_conv)

        elif isinstance(child, FeedbackLinearLayer):
            in_features = child.in_features
            out_features = child.out_features
            new_linear = nn.Linear(in_features, out_features)
            new_linear = new_linear.to(device)
            new_linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_linear.bias.data.copy_(child.bias.data)
            setattr(module, name, new_linear)
        else:
            convert_feedback_to_bp(child, device)