import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedbackConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, B, bias, stride, padding, dilation, groups):
        ctx.save_for_backward(input, B, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, B, bias = ctx.saved_tensors
        stride     = ctx.stride
        padding    = ctx.padding
        dilation   = ctx.dilation
        groups     = ctx.groups

        grad_input = torch.nn.grad.conv2d_input(input.shape, B, grad_output,
                                                  stride, padding, dilation, groups)

        grad_weight = torch.nn.grad.conv2d_weight(input, B.shape, grad_output,
                                                  stride, padding, dilation, groups)
        # bias가 있을 경우 기울기 계산
        grad_bias = grad_output.sum(dim=(0, 2, 3)) if bias is not None else None

        return grad_input, grad_weight, None, grad_bias, None, None, None, None

class FeedbackConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(FeedbackConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels // groups,
                                                kernel_size, kernel_size))
        self.register_buffer('B', self.weight.data.clone())
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, input):
        return FeedbackConvFunction.apply(input, self.weight, self.B, self.bias,
                                            self.stride, self.padding,
                                            self.dilation, self.groups)
    
    def scale_B(self):
        with torch.no_grad():
            weight_flat = self.weight.view(self.weight.size(0), -1)
            B_flat = self.B.view(self.B.size(0), -1)
            weight_norm = weight_flat.norm(p=2, dim=1, keepdim=True)
            B_norm = B_flat.norm(p=2, dim=1, keepdim=True)
            scaling = weight_norm / (B_norm + 1e-8)
            B_flat = B_flat * scaling
            self.B.copy_(B_flat.view_as(self.B))