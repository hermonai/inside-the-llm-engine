"""Ch 13 follow-up: how many GPU kernels does torch.compile emit for a chain of
n dependent additions on MPS? Run with TORCH_LOGS=output_code; the generated
source goes to stderr, and this script prints the result and the kernel count."""
import sys, torch
dev = torch.device("mps")
x0 = torch.zeros(1024, device=dev)
for n in (10, 1000):
    torch._dynamo.reset()
    def many(x, n=n):
        for _ in range(n): x = x + 1
        return x
    r = torch.compile(many)(x0); torch.mps.synchronize()
    print(n, bool(torch.equal(r, x0 + float(n))), flush=True)
