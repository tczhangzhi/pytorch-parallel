import torch

def _get_device_index(device, optional=False):
    if isinstance(device, torch.device):
        dev_type = device.type
        if dev_type != 'cuda':
            raise ValueError('Expected a cuda device, but got: {}'.format(device))
        device_idx = device.index
    else:
        device_idx = device
    if device_idx is None:
        if optional:
            # default cuda device index
            return torch.cuda.current_device()
        else:
            raise ValueError('Expected a cuda device with sepecified index or '
                             'an integer, but got: {device}'.format(device=device))
    return device_idx