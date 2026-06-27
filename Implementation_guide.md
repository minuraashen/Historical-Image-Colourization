# Putting RAM (All-in-One Image Restoration) on an Edge NPU — End-to-End Guide

A practical, plain-English walkthrough to take the **RAM** restoration model from a PyTorch checkpoint to a model that runs (and is *measured*) on a real Snapdragon NPU — **without owning a phone**, using free cloud tooling. Written as a feasibility + measurement harness you can reuse for your research.

---

## 0. The big picture in one paragraph

You will (1) get RAM running normally in PyTorch, (2) convert it into an export-friendly form, (3) upload it to **Qualcomm AI Hub**, which compiles it for a real Snapdragon chip and runs it on an actual device in the cloud, reporting latency, memory, and — most importantly — **which layers ran on the NPU vs. fell back to CPU**, (4) quantize it to INT8/W8A16 and re-measure, (5) fix any operations the NPU rejects, (6) add tiling so full-size images fit in the NPU's small memory, and (7) wrap it in a minimal app. No phone required until the very last optional step.

**Why Qualcomm AI Hub and not an emulator:** the Android emulator and iOS simulator do **not** give you a real NPU — they fall back to CPU, so they cannot verify NPU execution. AI Hub provisions *real* Snapdragon devices in the cloud, for free, and is the correct way to verify on-device behavior.

---

## 1. What "can it run on NPU" actually depends on

Three things, in order of importance. Keep these in mind the whole way through:

1. **Operator support** — every layer (conv, attention, normalization, activation, reshape) must have an NPU implementation. One unsupported op forces a CPU fallback for that part, which kills latency.
2. **Static shapes** — NPUs want fixed input sizes. Dynamic shapes, variable loops, and data-dependent control flow break compilation.
3. **Memory** — NPUs have small fast memory. Big activation maps (high-res images) must be tiled.

RAM is a good candidate because it is a **convolution + attention U-shaped network with no diffusion sampling, no VAE, and no text encoder** — so it avoids the worst NPU-hostile parts of Stable-Diffusion-based models. Its degradation-adaptation module (gated sub-latent reweighting) is the part most likely to contain an awkward op; the profiling step will tell you for sure.

---

## 2. Accounts and tools you need (all free)

- **Python 3.10** environment (conda or venv).
- **PyTorch** (CPU build is fine for export; a GPU or free Colab/Kaggle helps for the baseline run).
- **A Qualcomm AI Hub account** — sign up at aihub.qualcomm.com, then get your API token from the settings page.
- The **`qai-hub`** Python package (the client) and optionally **`qai-hub-models`** (their model zoo + helpers).
- The **RAM code + checkpoint.** RAM = *Restore Anything Model via Efficient Degradation Adaptation* (Ren et al., arXiv:2407.13372). If the official RAM release is incomplete, the same group's follow-up **AnyIR** (arXiv:2504.14249, code at github.com/Amazingren/AnyIR) is a near-identical lightweight all-in-one restorer with a maintained repo — a safe fallback to build the pipeline on.

> Naming warning: a *different* paper, *Restore Anything with Masks* (arXiv:2409.19403), is also abbreviated "RAM." Make sure you use the **Efficient Degradation Adaptation** one (2407.13372).

Install:

```bash
conda create -n ram-npu python=3.10 -y
conda activate ram-npu
pip install torch torchvision pillow numpy
pip install qai-hub qai-hub-models
qai-hub configure --api_token YOUR_TOKEN_HERE
```

---

## 3. Phase 1 — Get RAM working in plain PyTorch (your laptop or Colab)

Before any NPU work, prove the model runs and restores images correctly on a normal machine.

1. Clone the RAM (or AnyIR) repo and follow its README to install dependencies and download the pretrained checkpoint.
2. Load the model in `eval()` mode.
3. Feed it one degraded test image (noise / blur / low-light / haze, depending on what the checkpoint supports) and save the output.
4. Confirm the restoration looks right. This is your **reference output** — every later step is checked against it.

```python
import torch
from PIL import Image
import torchvision.transforms as T

model = build_ram_model()                 # per the repo's instructions
model.load_state_dict(torch.load("ram.pth", map_location="cpu"))
model.eval()

img = Image.open("degraded.png").convert("RGB")
x = T.ToTensor()(img).unsqueeze(0)        # shape [1, 3, H, W]
with torch.no_grad():
    y = model(x)
T.ToPILImage()(y.squeeze(0).clamp(0, 1)).save("restored_reference.png")
```

If the model returns a dictionary or multiple outputs, note that — you'll simplify it in the next phase.

---

## 4. Phase 2 — Make RAM export-friendly

NPUs compile a *static graph*, so you must remove anything dynamic and fix the input size.

**a) Fix the input shape.** Pick a tile size you'll process, e.g. `256×256` (a good NPU-friendly size). You will feed the network fixed `[1, 3, 256, 256]` tensors and handle full images by tiling later.

**b) Make the model return a single tensor, not a dict.** AI Hub doesn't handle dictionary outputs well. Wrap the model:

```python
class ExportWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        out = self.net(x)
        if isinstance(out, dict):
            out = out["output"]            # use the real key from the repo
        return out                          # single tensor [1, 3, 256, 256]
```

Also set `model.config.return_dict = False` if the model exposes that flag.

**c) Avoid NPU-hostile ops where you can.** Common offenders in restoration models: dynamic shapes, `einsum` with unusual patterns, certain `nn.functional.interpolate` modes, and exotic normalizations. You don't need to fix these blindly — compile first and let the profiler tell you what breaks (Phase 4). But knowing the usual suspects speeds up debugging.

**d) Trace the model** into a static graph:

```python
wrapped = ExportWrapper(model).eval()
example = torch.rand(1, 3, 256, 256)
traced = torch.jit.trace(wrapped, example)
traced.save("ram_traced.pt")
```

If tracing throws errors, they almost always point at dynamic control flow — fix those first.

---

## 5. Phase 3 — Compile and profile on a real NPU (the key step)

This is where you get your answer. AI Hub compiles your traced model for a chosen Snapdragon device, runs it on the physical chip in the cloud, and reports real numbers.

```python
import qai_hub as hub

# 1. Compile the traced PyTorch model for a real device + runtime
compile_job = hub.submit_compile_job(
    model="ram_traced.pt",
    input_specs={"x": (1, 3, 256, 256)},
    device=hub.Device("Samsung Galaxy S24 (Family)"),   # Snapdragon 8 Gen 3
    options="--target_runtime tflite",                  # tflite or qnn_context_binary
)
compiled = compile_job.get_target_model()

# 2. Profile it on the real device (runs ~100 iterations on physical hardware)
profile_job = hub.submit_profile_job(
    model=compiled,
    device=hub.Device("Samsung Galaxy S24 (Family)"),
)
```

When the profile job finishes, the AI Hub web dashboard shows you:

- **Inference latency** (milliseconds) — your headline number.
- **Peak memory usage** — tells you whether the tile size fits.
- **Compute-unit breakdown** — *the critical one.* It lists each layer and whether it ran on **NPU (HTP)**, **GPU**, or **CPU**. If everything says NPU/HTP, you've succeeded. If chunks say CPU, those are your fallback ops to fix in Phase 5.

Useful devices to try: `"Samsung Galaxy S24 (Family)"` (8 Gen 3), `"Samsung Galaxy S23 (Family)"` (8 Gen 2), and `"Snapdragon 8 Elite QRD"` (latest). Profiling several gives you a fair latency picture.

> Note: a floating-point (FP32/FP16) model may be forced onto CPU on devices whose NPU only accepts integer math. That's expected — the NPU win comes after quantization (Phase 6). Profile the FP model first anyway, as a correctness and baseline check.

---

## 6. Phase 4 — Check numerical correctness on-device

Latency means nothing if the output is wrong. Run an **inference job**, which executes your model once on the real device with your input and returns the actual output so you can compare it to your Phase-1 reference.

```python
import numpy as np

sample = np.random.rand(1, 3, 256, 256).astype(np.float32)
inference_job = hub.submit_inference_job(
    model=compiled,
    device=hub.Device("Samsung Galaxy S24 (Family)"),
    inputs={"x": [sample]},
)
on_device_out = inference_job.download_output_data()
# Compare on_device_out against your PyTorch output for the same input.
```

A small difference is fine; a large one means the compile/quantization changed behavior and needs investigation.

---

## 7. Phase 5 — Fix operator fallbacks (iterate)

For each layer the profiler showed running on CPU:

1. Identify the operation (the dashboard names it).
2. Replace it with an NPU-friendly equivalent. Typical fixes:
   - swap an unusual activation (e.g., exotic GELU variant) for a standard `ReLU`/`GELU`,
   - replace a dynamic `interpolate` with a fixed-scale `nn.Upsample` or a transposed conv,
   - rewrite an attention block so its reshapes use static shapes,
   - move any data-dependent logic out of the traced forward pass.
3. Re-trace, re-compile, re-profile. Repeat until the graph stays on the NPU.

This loop is the single most valuable thing you'll learn — it directly tells you which building blocks are safe for your future distilled student.

---

## 8. Phase 6 — Quantize to INT8 / W8A16

NPUs are fastest (and sometimes only work) with integer math. AI Hub can quantize during compilation.

```python
compile_job = hub.submit_compile_job(
    model="ram_traced.pt",
    input_specs={"x": (1, 3, 256, 256)},
    device=hub.Device("Samsung Galaxy S24 (Family)"),
    options="--target_runtime tflite --quantize_full_type w8a16",
)
```

- Start with **W8A16** (8-bit weights, 16-bit activations) — this is what the proven edge models (Edge-SD-SR, NanoSD) use, and it's far more accurate than INT8-everything for image work.
- Provide a small **calibration set** (a few dozen representative images) so quantization picks good ranges — the repo/AI Hub docs show how to pass calibration data.
- Re-run the inference job and compare output quality to your reference. If quality drops too much, keep more layers in 16-bit (mixed precision).
- Re-profile: quantization is usually where the latency drops sharply and the layers finally land on the NPU.

For finer control or quantization-aware training later, Qualcomm's **AIMET** library is the free tool, but AI Hub's built-in quantization is enough to start.

---

## 9. Phase 7 — Tiling for real, full-size images

NPU memory is small, so you process big images in patches:

1. Split the input image into overlapping tiles of your fixed size (e.g., `256×256` with a 16–32px overlap).
2. Run each tile through the model.
3. Blend the overlaps (feather/average) when stitching back, to avoid visible seams.

This runs on the host (Python/Kotlin) around the NPU calls. It's the same trick Edge-SD-SR uses to upscale a full image as a sequence of patch evaluations.

```python
def restore_tiled(image, run_tile, tile=256, overlap=32):
    # pseudo-code: loop over tiles, run_tile() each, feather-blend into output
    ...
```

---

## 10. Phase 8 — The app (two realistic routes)

**Route A — Web app (recommended first).** Fastest path to a working demo and a thesis/portfolio artifact.

- Backend (FastAPI/Flask) runs RAM in PyTorch on a server (CPU or a small GPU).
- Frontend: upload a degraded image → see the restored result side by side, with a download button.
- For the "edge" story, you **don't run the NPU in the web app** — instead you display the **AI Hub measurements** (latency, memory, NPU compute-unit breakdown) as your on-device feasibility evidence. This cleanly separates "the model works" (web demo) from "it fits on an NPU" (AI Hub numbers).
- Effort: a few days. This is the highest value-for-time option.

**Route B — Real Android app (optional, later).** The genuine on-device demo.

- Bundle the compiled `.tflite` (LiteRT) or `.bin` (QNN context binary) you downloaded from AI Hub.
- Use the **LiteRT** runtime (formerly TensorFlow Lite) with the NNAPI/QNN delegate, or the **Qualcomm AI Runtime (QNN)** directly, inside an Android Studio project.
- Do the tiling in Kotlin around the model calls.
- You can build and unit-test most of this in Android Studio, but the *final* NPU demo needs a real Snapdragon phone — borrow one for a day at the end, after AI Hub has already proven it works.
- Effort: 1–2 weeks if new to Android.

**Suggested order:** Build Route A now. Only attempt Route B if you specifically need a live on-phone demo for your defense.

---

## 11. Realistic timeline and scope control

| Phase | What you get | Rough effort |
|---|---|---|
| 1 — PyTorch baseline | RAM restores images on your machine | 1–2 days |
| 2 — Export-friendly | Traced static model | 1–2 days |
| 3–4 — Compile + profile + correctness | **Real NPU latency/memory + go/no-go answer** | 2–4 days |
| 5 — Fix fallbacks | Graph stays on NPU | 2–5 days (the variable part) |
| 6 — Quantize | INT8/W8A16, big latency drop | 2–3 days |
| 7 — Tiling | Full-size images | 1–2 days |
| 8 — Web demo (Route A) | Shareable app | 2–4 days |

**Total for a solid feasibility harness + demo: ~3–4 weeks.** Timebox it. The moment Phases 3–6 are done you have everything the research needs; resist polishing the app beyond a clean demo.

---

## 12. How this directly feeds the research (so it isn't wasted)

- The **compile → quantize → profile → tile pipeline is your evaluation harness** for the thesis. You'll reuse it unchanged to measure your distilled student.
- The **operator-fallback findings tell you which building blocks are NPU-safe**, which is exactly the evidence you need to justify your student-architecture choices in the paper ("why this backbone").
- The **real on-device latency/memory numbers** are the credibility requirement for a CVPR/ICCV submission in the edge space — you'll be able to produce them on demand.
- RAM becomes your **non-generative baseline**: "here's the best efficient feed-forward all-in-one model on an NPU; here's the quality gap our flow-distilled generative student closes at similar cost."

**One honest limitation:** RAM is non-generative, so this exercise validates the *deployment* half of your thesis, not the *generative-quality-on-edge* half. That's fine — it's the right thing to de-risk first, and it gives you a baseline and a working measurement rig before you train anything.

---

## 13. Quick reference — the minimal happy path

```text
sign up at aihub.qualcomm.com  →  get token  →  qai-hub configure
run RAM in PyTorch             →  save reference output
wrap (single tensor out) + fix input size 256x256  →  torch.jit.trace
hub.submit_compile_job(... device=S24 ... tflite)
hub.submit_profile_job(...)    →  read latency / memory / NPU-vs-CPU map
hub.submit_inference_job(...)  →  check output matches reference
re-compile with --quantize_full_type w8a16 + calibration  →  re-profile
fix any CPU-fallback ops, repeat
add tiling for full images
build a small web demo; show AI Hub numbers as edge evidence
```

If any step blocks you, the AI Hub docs (app.aihub.qualcomm.com/docs) and their Slack are responsive, and the `qai-hub-models` repo has working end-to-end examples (e.g. their `aotgan` image-inpainting model is a close structural cousin to a restoration U-Net and a good reference to copy).
