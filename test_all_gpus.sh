for i in $(seq 0 7); do
  echo "=============================="
  echo "Testing GPU $i"
  echo "=============================="

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$i python - <<'PY'
import torch, time, sys

try:
    assert torch.cuda.is_available(), "CUDA not available"
    assert torch.cuda.device_count() == 1, f"Expected 1 visible GPU, got {torch.cuda.device_count()}"

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    total_mb = props.total_memory // (1024**2)

    print(f"Testing: {name}")
    print(f"Total memory: {total_mb} MiB")

    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)

    torch.cuda.synchronize()
    t0 = time.time()

    iters = 0
    while time.time() - t0 < 10:
        c = a @ b
        iters += 1

    torch.cuda.synchronize()
    print(f"OK - iterations: {iters}, elapsed: {time.time() - t0:.2f}s")
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MiB")

except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
PY

  rc=$?
  if [ $rc -ne 0 ]; then
    echo "GPU $i FAILED"
  else
    echo "GPU $i PASSED"
  fi
done