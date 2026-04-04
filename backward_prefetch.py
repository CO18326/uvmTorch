import torch
import ctypes

my_lib = ctypes.CDLL("./prefetch_async.so")
my_lib.prefetch_memory.argtypes = [ctypes.c_ulong, ctypes.c_ulong, ctypes.c_int, ctypes.c_void_p]
my_lib.prefetch_memory.restype = ctypes.c_int
my_lib.pin_memory_hint.argtypes = [ctypes.c_ulong, ctypes.c_ulong, ctypes.c_int]
my_lib.pin_memory_hint.restype = ctypes.c_int
#my_lib.prefetch_memory_batch.argtypes = [ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_int, ctypes.c_void_p]
#my_lib.prefetch_memory_batch.restype = ctypes.c_int
my_lib.cuda_malloc.argtypes=[ctypes.c_ulong]
my_lib.cuda_malloc.restype=ctypes.c_int
try:
    my_lib.print_first_byte.restype = ctypes.c_int
except Exception:
    pass


def isinstance_namedtuple(obj: object) -> bool:
    """
    Is this an instance of namedtuple/NamedTuple?
    From: https://stackoverflow.com/a/62692640

    Args:
        obj (object): An object.

    Returns:
        bool: True if namedtuple/NamedTuple else False.
    """
    return isinstance(obj, tuple) and hasattr(obj, '_asdict') and hasattr(obj, '_fields')



def is_builtin_type(obj):
    # https://stackoverflow.com/a/17795199
    return obj.__class__.__module__ == '__builtin__' or obj.__class__.__module__ == "builtins"




def apply_to_tensors_only(function, value, warning_msg_fn=None):
    """
    Apply `function` to every Tensor in `value`.

    Args:
        functional: The function class to apply.
        value (Any): Target object to apply `function` to.

    Returns:
        Any: Output of `function`.
    """
    if isinstance(value, (tuple, list)):
        touched_outputs = []
        for elem in value:
            touched_output = apply_to_tensors_only(function, elem)
            touched_outputs.append(touched_output)

        if isinstance_namedtuple(value):
            # namedtuples require a slightly different syntax.
            return value.__class__(*touched_outputs)

        return value.__class__(touched_outputs)
    elif isinstance(value, dict):
        # apply inplace to avoid recreating dict inherited objects
        for key in value.keys():
            value[key] = apply_to_tensors_only(function, value[key])
        return value

    elif isinstance(value, torch.Tensor):
        # this also applies to torch.Tensor's subclasses like torch.nn.parameter.Parameter
        touched_output = function(value)

        # restore zero param attributes if those get stripped by `backward_function`
       

        return touched_output
    else:
        if not is_builtin_type(value):
            global warned
            if warning_msg_fn and not warned:
                print("Error")
                warned = True
        return value



def _pre_backward_module_hook(module, inputs, output):
    #return apply_to_tensors_only(module.pre_bwd_fn.apply,
                                         #output,inputs)
    
    return module.pre_bwd_fn.apply(output,inputs)







