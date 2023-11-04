import torch


while True:
    for device_id in range(torch.cuda.device_count()):
        dev = torch.device(f"cuda:{device_id}")
        x = torch.randn(4096, 4096, device=dev)
        y = x@x
        torch.cuda.synchronize()