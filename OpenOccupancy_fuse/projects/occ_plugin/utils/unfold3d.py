import torch
import torch.nn as nn
import torch.nn.functional as F

class Unfold3D(nn.Module):
    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
        super(Unfold3D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        d_in, h_in, w_in = x.shape[-3], x.shape[-2], x.shape[-1]
        d_out = (d_in + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        h_out = (h_in + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        w_out = (w_in + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        
        x_padded = torch.nn.functional.pad(x, (self.padding,) * 6)
        # [B, C, D, H, W]
        patches = x.new_empty((x.shape[0], x.shape[1], self.kernel_size, self.kernel_size, self.kernel_size, d_out, h_out, w_out))
        
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                for k in range(self.kernel_size):
                    start_d = i * self.dilation
                    start_h = j * self.dilation
                    start_w = k * self.dilation

                    end_d = start_d + self.stride * d_out
                    end_h = start_h + self.stride * h_out
                    end_w = start_w + self.stride * w_out

                    patches[:, :, i, j, k] = x_padded[:, :, start_d: end_d: self.stride,
                                                            start_h: end_h: self.stride,
                                                            start_w: end_w: self.stride]

        patches = patches.view(x.shape[0], -1, d_out, h_out, w_out)
        return patches


if __name__ == "__main__":
    # Example usage:
    input_tensor = torch.randn(1, 32, 30, 30, 30)  # Batch size: 1, Channels: 1, Depth/Height/Width: 30
    unfold = Unfold3D(kernel_size=3, stride=2, padding=1, dilation=1)
    patches = unfold(input_tensor)
    print(patches.shape)  # Expected shape: (1, C*k^3, new_D, new_H, new_W)
