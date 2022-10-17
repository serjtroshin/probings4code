import torch


def to_cpu(obj):
    if isinstance(obj, (dict,)):
        for key, value in obj.items():
            obj[key] = to_cpu(value)
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.to("cpu")
    elif isinstance(obj, (list, tuple)):
        return [to_cpu(value) for value in obj]
    else:
        return obj
