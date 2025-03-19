import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedbackLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, B, bias):
        ctx.save_for_backward(input, B, bias)
        output = F.linear(input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, B, bias = ctx.saved_tensors

        grad_input = grad_output.matmul(B)

        grad_weight = grad_output.t().matmul(input)

        grad_bias = grad_output.sum(dim=0) if bias is not None else None

        return grad_input, grad_weight, None, grad_bias

class FeedbackLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(FeedbackLinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))

        self.register_buffer('B', self.weight.data.clone())
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, input):
        return FeedbackLinearFunction.apply(input, self.weight, self.B, self.bias)
    
    def scale_B(self):
        with torch.no_grad():
            weight_flat = self.weight.view(self.weight.size(0), -1)
            B_flat = self.B.view(self.B.size(0), -1)
            weight_norm = weight_flat.norm(p=2, dim=1, keepdim=True)
            B_norm = B_flat.norm(p=2, dim=1, keepdim=True)
            scaling = weight_norm / (B_norm + 1e-8)
            B_flat = B_flat * scaling
            self.B.copy_(B_flat.view_as(self.B))