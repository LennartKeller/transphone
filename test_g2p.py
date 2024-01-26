from torch.profiler import ProfilerActivity, profile, record_function

from transphone.g2p import read_g2p

g2p_model_cpu = read_g2p(device="cpu")
g2p_model_onnx = read_g2p(device="onnx")


with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof_cpu:
    with record_function("model_inference"):
        print(g2p_model_cpu.inference_batch("hallo", "deu"))

print("CPU")
print(prof_cpu.key_averages().table(sort_by="cpu_time_total", row_limit=20))


with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof_onnx:
    with record_function("model_inference"):
        print(g2p_model_onnx.inference_batch("hallo", "deu"))

print("ONNX")
print(prof_onnx.key_averages().table(sort_by="cpu_time_total", row_limit=20))



