import torch
import time

N = 24 * 1024
# N = 32 * 1024

x = torch.rand([N, N], dtype=torch.float64, device='cuda')
w = torch.rand([3, 3], dtype=torch.float64, device='cuda')

torch.cuda.reset_peak_memory_stats(torch.device("cuda"))
max_memory_allocated = torch.cuda.max_memory_allocated(torch.device("cuda"))
# 开始 计算之前：
print(f"开始 计算之前 最大显存占用: {max_memory_allocated / (1024 ** 3)} GB")

# 测 最大显存占用
torch.cuda.reset_peak_memory_stats(torch.device("cuda"))
y = torch.fft.ifft2(torch.fft.fft2(x) * torch.fft.fft2(w, s=x.size())).real
print(y, y.shape)
max_memory_allocated = torch.cuda.max_memory_allocated(torch.device("cuda"))
print(f"FFT 最大显存占用: {max_memory_allocated / (1024 ** 3)} GB")


torch.cuda.synchronize()
begin = time.perf_counter()
for i in range(10): 
    y = torch.fft.ifft2(torch.fft.fft2(x) * torch.fft.fft2(w, s=x.size())).real
torch.cuda.synchronize()
end = time.perf_counter()

print('FFT cost:', (end - begin) * 1000 / 10)