import torch

print("--- Test 1: Pure FP32 ---")
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print("Result:", s.item())


print("\n--- Test 2: Pure FP16 ---")
s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print("Result:", s.item())


print("\n--- Test 3: FP32 Accumulator, FP16 additions (implicit cast) ---")
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print("Result:", s.item())


print("\n--- Test 4: FP32 Accumulator, FP16 additions (explicit cast) ---")
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print("Result:", s.item())
