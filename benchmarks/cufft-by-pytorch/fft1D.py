import torch
import time
import sys

args = sys.argv

input_size = 512 * 1024 * 1024 
kernel_size = int(args[1])

print("k=", kernel_size)

x = torch.rand([input_size], dtype=torch.float64, device='cuda')
w = torch.rand([kernel_size], dtype=torch.float64, device='cuda')

torch.cuda.reset_peak_memory_stats(torch.device("cuda"))
max_memory_allocated = torch.cuda.max_memory_allocated(torch.device("cuda"))
# 开始 计算之前：
print(f"开始 计算之前 最大显存占用: {max_memory_allocated / (1024 ** 3)} GB")

# 测 最大显存占用
torch.cuda.reset_peak_memory_stats(torch.device("cuda"))
y = torch.fft.ifft(torch.fft.fft(x) * torch.fft.fft(w, x.size(0))).real
print(y, y.shape)
max_memory_allocated = torch.cuda.max_memory_allocated(torch.device("cuda"))
print(f"FFT 最大显存占用: {max_memory_allocated / (1024 ** 3)} GB")


torch.cuda.synchronize()
begin = time.perf_counter()
for i in range(10): 
    y = torch.fft.ifft(torch.fft.fft(x) * torch.fft.fft(w, x.size(0))).real
torch.cuda.synchronize()
end = time.perf_counter()

time = (end - begin) * 1000 / 10
print('FFT cost: [ms]', time)

print("GStencil/s = ", input_size * ((kernel_size - 1) / 2) * 1 / time / 1e6)