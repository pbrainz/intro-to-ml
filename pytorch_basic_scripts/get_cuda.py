import torch

print("\n\n Checking Cuda... \n\n")

print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device: {torch.cuda.current_device()}")
print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")


print("\n\n Testing CPU and Cuda... \n\n")
# Creating a test tensor
x = torch.randint(1, 100, (100, 100))

# Checking the device name:
# Should return 'cpu' by default
print(x.device)

# Applying tensor operation
res_cpu = x ** 2

# Transferring tensor to GPU
x = x.to(torch.device('cuda'))

# Checking the device name:
# Should return 'cuda:0'
print(x.device)

# Applying same tensor operation
res_gpu = x ** 2

# Checking the equality
# of the two results
assert torch.equal(res_cpu, res_gpu.cpu())
