import torch
import numpy as np

def move_tensor_to_gpu(tensor: torch.Tensor) -> torch.Tensor:
    if torch.cuda.is_available():
        tensor = tensor.to('cuda')
        return tensor

# Initializing
data = [[1, 2], [3, 4]]
x1 = torch.tensor(data)
print(f"x1: \n{x1}")

x2 = torch.randint_like(x1, high=100, dtype=torch.int64)
print(f"x2: \n {x2}")

# Attributes
print(f"Shape of tensor: {x2.shape}")
print(f"Datatype of tensor: {x2.dtype}")
print(f"Device tensor is stored on: {x2.device}")

# Joining tensors
t1 = torch.cat([x1, x2], dim=1)
print(f"concatenation: \n {t1}")

# Arithemetic operations
t2 = x1 @ x2.T
print(f"matrix multiplication: \n {t2}")

t3 = x1 * x2
print(f"element-wise product: \n {t3}")

# Single element tensors
agg = t3.sum()
agg_item = agg.item()
print(f"agg item: \n {agg_item}", type(agg_item))

# In place operations
x1.add_(5)
print(f"in place op: \n {x1}")


# Tensor to Numpy array
# Share same memory locations
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# Change in one reflects in the other
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# Numpy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 4, out=n)
print(f"t: {t}")
print(f"n: {n}")