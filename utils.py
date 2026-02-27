import torch

def is_torch_mps_available():
    try:
        import torch
        torch.device('mps')
        return True
    except RuntimeError:
        return False

def set_device(gpu=-1):
    """Setup the device.

    Args:
        gpu (int, optional): num of GPU to use. Defaults to -1 (not use GPU, i.e., use CPU).
    """
    if gpu >= 0:
        if torch.cuda.is_available():
            device = torch.device("cuda:" + str(gpu))
        elif is_torch_mps_available():
            device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device