import torch
import torch.nn.functional as F
import kornia.losses

def L2_loss(y_true, y_pred):
    cri = torch.nn.MSELoss()
    loss = cri(y_pred, y_true)
    return loss 

def Edge_loss(y_true, y_pred):
    loss = 0
    
    # Sobel kernel
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=y_true.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=y_true.device).view(1, 1, 3, 3)

    # Compute edges by convolving the input
    edge_true = 0
    edge_pred = 0
    for c in range(y_true.shape[1]):
        edge_true_x = F.conv2d(y_true[:, c:c+1], kernel_x, padding=1)
        edge_true_y = F.conv2d(y_true[:, c:c+1], kernel_y, padding=1)
        edge_pred_x = F.conv2d(y_pred[:, c:c+1], kernel_x, padding=1)
        edge_pred_y = F.conv2d(y_pred[:, c:c+1], kernel_y, padding=1)
        edge_true += torch.sqrt(edge_true_x**2 + edge_true_y**2 + 1e-6)
        edge_pred += torch.sqrt(edge_pred_x**2 + edge_pred_y**2 + 1e-6)
    edge_true /= y_true.shape[1]
    edge_pred /= y_pred.shape[1]
    
    # Compute L1 loss
    loss = F.l1_loss(edge_pred, edge_true)
    return loss


def _gaussian_kernel(channels: int, device=None, dtype=None):
    """ Fixed 5x5 Gaussian kernel with standard deviation ~1. Performs depthwise convolution for each channel. """
    k = torch.tensor([
        [1.,  4.,  6.,  4., 1.],
        [4., 16., 24., 16., 4.],
        [6., 24., 36., 24., 6.],
        [4., 16., 24., 16., 4.],
        [1.,  4.,  6.,  4., 1.]
    ], device=device, dtype=dtype)
    k = (k / k.sum()).to(dtype)
    kernel = k.expand(channels, 1, 5, 5).contiguous()
    return kernel

def _gaussian_blur(x):
    """ Apply Gaussian blur to input (depthwise convolution). """
    n, c, h, w = x.shape
    kernel = _gaussian_kernel(c, device=x.device, dtype=x.dtype)
    padding = 2  # 5x5 kernel -> same padding
    return F.conv2d(x, kernel, bias=None, stride=1, padding=padding, groups=c)

def _downsample(x):
    """ Downsample with stride=2 after Gaussian blur (equivalent to blur then take even points). """
    x_blur = _gaussian_blur(x)
    return x_blur[:, :, ::2, ::2]

def _upsample(x, size):
    """ Bilinear upsampling to specified size, followed by Gaussian blur for pyramid stability. """
    x_up = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    x_up = _gaussian_blur(x_up)
    return x_up

def _build_laplacian_pyramid(x, levels: int):
    gaussians = [x]
    for _ in range(1, levels):
        gaussians.append(_downsample(gaussians[-1]))
    laps = []
    for i in range(levels - 1):
        current = gaussians[i]
        next_g = gaussians[i + 1]
        up = _upsample(next_g, size=current.shape[-2:])
        laps.append(current - up)
    laps.append(gaussians[-1])
    return laps

def Laplacian_loss(y_true, y_pred, levels=3, weights=None, reduction="mean"):
    assert y_true.dim() == 4 and y_pred.dim() == 4, "Inputs must be [B, C, H, W]."
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape."
    assert levels >= 1, "levels must be >= 1"
    if weights is None:
        weights = [0.5 ** i for i in range(levels)]
    else:
        if len(weights) < levels:
            tail = [weights[-1] * (0.5 ** i) for i in range(1, levels - len(weights) + 1)]
            weights = list(weights) + tail
        elif len(weights) > levels:
            weights = list(weights[:levels])
    pyr_true = _build_laplacian_pyramid(y_true, levels)
    pyr_pred = _build_laplacian_pyramid(y_pred, levels)
    losses = []
    for w, t, p in zip(weights, pyr_true, pyr_pred):
        l = torch.abs(t - p)
        if reduction == "mean":
            l = l.mean()
        elif reduction == "sum":
            l = l.sum()
        elif reduction == "none":
            pass
        else:
            raise ValueError("reduction must be 'mean' | 'sum' | 'none'")
        losses.append(w * l)
    if reduction == "none":
        return losses
    else:
        return sum(losses)

def SSIM_loss(y_true, y_pred, window_size=11, sigma=1.5, data_range=1.0, reduction="mean", eps=1e-6):
    ssim_loss = kornia.losses.SSIMLoss(
        window_size=window_size,
        reduction=reduction,
        max_val=data_range,
    )
    return ssim_loss(y_pred, y_true)

def Design_loss(pred, target, epoch):
    L2_Loss = L2_loss(target, pred)
    Laplacian_Loss = Laplacian_loss(target, pred)
    Edge_Loss = Edge_loss(target, pred)
    SSIM_Loss = SSIM_loss(target, pred)
    if epoch < 10:
        lambda_L2, lambda_lap, lambda_edge, lambda_ssim = 1.0, 0.00, 0.00, 0.000
        Loss = lambda_L2 * L2_Loss + lambda_lap * Laplacian_Loss + lambda_edge * Edge_Loss + lambda_ssim * SSIM_Loss
    else:
        lambda_L2, lambda_lap, lambda_edge, lambda_ssim = 1.0, 0.01, 0.005, 0.001
        Loss = lambda_L2 * L2_Loss + lambda_lap * Laplacian_Loss + lambda_edge * Edge_Loss + lambda_ssim * SSIM_Loss
    return Loss, L2_Loss
