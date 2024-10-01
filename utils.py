import torch

def to_tensor_batched(x: any, batch_dims: int) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    if x.dim() < batch_dims:
        x = x.unsqueeze(0)

    return x
