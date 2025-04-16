import torch
import functools

# Save original torch.cat function
original_cat = torch.cat

# Helper function to synchronize devices
def sync_tensors_to_same_device(tensor_list):
    """Ensure all tensors are on the same device"""
    if not tensor_list:
        return tensor_list
    
    # Find the target device (we choose the device of the first non-None tensor)
    target_device = None
    for t in tensor_list:
        if t is not None and hasattr(t, 'device'):
            target_device = t.device
            break
    
    if target_device is None:
        return tensor_list
    
    # Move all tensors to the same device
    result = []
    for t in tensor_list:
        if t is not None and hasattr(t, 'device') and t.device != target_device:
            # print(f"Moving tensor from {t.device} to {target_device}")
            result.append(t.to(target_device))
        else:
            result.append(t)
    
    return result

# Create a new torch.cat implementation that ensures all tensors are on the same device
@functools.wraps(original_cat)
def patched_cat(tensors, *args, **kwargs):
    # Ensure all tensors are on the same device
    synced_tensors = sync_tensors_to_same_device(tensors)
    return original_cat(synced_tensors, *args, **kwargs)

# Apply the monkey patch
def apply_patch():
    torch.cat = patched_cat
    print("Applied torch.cat patch, will automatically synchronize tensors on different devices")

# Restore original function
def remove_patch():
    torch.cat = original_cat
    print("Removed torch.cat patch, restored original behavior")
