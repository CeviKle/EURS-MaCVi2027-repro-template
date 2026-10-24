# EURS @ MaCVi WACV 2027: reproduction repository template

Fork or copy this repository for your final hand-in. The organizers **re-run your inference** on the testing-phase images
with the weights of the submission you report, score it with the leaderboard evaluator and compare. Only teams whose result
reproduces are ranked, in **both** tracks (F = 0.70 x Track 2 + 0.30 x Track 1), so hand in one repository (or one
folder per track) for each track you entered.

**Deadline: October 28, 2026, 23:59 AoE**, together with your factsheet (see the factsheet template).

## What to put in the repository

```
METHOD.md            one page: approach, training data, ALL external data / pretrained models, anything non-standard at inference
repro.yaml           team, track, submission_id (your best-scoring TESTING-PHASE submission in this track), contact
requirements.txt     ONLY extra pip packages the base image does not have, pinned (name==x.y.z); empty if you need none
src/                 your model and utility code, importable, no absolute paths
run.py               main(model_dir, input_dir, output_dir, device)       <- both tracks (see below)
export_onnx.py       main(model_dir, out_onnx)                            <- Track 2 only
weights/             your checkpoint, or weights_link.yaml with a public download link (see below)
repro_check.py       this template's checker: run it before you hand in
```

* `run.py` reads the degraded images in `input_dir` and writes `output_dir/restored/<stem>.png` (RGB) and
  `output_dir/seg_masks/<stem>.png` (single channel, labels 0 to 4), resized to the scoring size: longest side at most 1024 px,
  aspect ratio kept. **Every output must come from its own input image only.** `model_dir` is your `weights/` folder.
  `run.py` here is the organizer baseline: replace the model, keep the signature and the output format.
* `export_onnx.py` (Track 2) regenerates the ONNX file you submitted from your weights: static input (1, 3, H, W) in [0, 1],
  outputs `restored_image` and `seg_logits`, at most 12 GFLOPs and 1.0 M parameters. We compare it with your leaderboard ONNX.
* **Training code is not needed.** A link to it in `METHOD.md` is welcome.
* **No network at run time**, so weights cannot be downloaded by your code. Make inference deterministic (fixed seeds,
  `model.eval()`, no test-time randomness). Time limit 90 minutes on one RTX 3090-class GPU.

## Weights: commit them, or give a public link (one way, not both)

**Committed:** small models go in `weights/`. Check that git really tracks them (`git ls-files weights`); delete `weights/weights_link.yaml`.

**Link:** larger models go behind a public link described in `weights/weights_link.yaml` (`url:`, `sha256:` of the file, and optionally
`filename:`); give the same link and hash in your factsheet. We download it **once, before the run**, verify the hash and save it in `weights/`
under the name the link's server gives it, or under `filename:` if you set it (recommended, so your `run.py` knows the exact name). A `.zip` or
`.tar.gz` is unpacked into `weights/` keeping its inner folders. Accepted: direct links, GitHub releases, Hugging Face, Dropbox, Google Drive
**file** links shared with "anyone with the link"; one file or one archive. Not accepted: Drive folders, OneDrive, links that need a login, Git LFS.
Keep the link working at least until the final results are announced (October 30, 2026). Do not commit the downloaded file to a link-based repository.
To test a link-based repository yourself, fetch the file into `weights/` first with `python repro_check.py . --download-weights`, then run the checks.

The repository must be **public** (or readable by the organizers) and pinned: give the **commit hash** in the factsheet and do not
change that commit afterwards.

## Reproduce your own pipeline exactly

We compare our re-run with the submission you uploaded to the leaderboard, so `run.py` must do **exactly what your own inference script did** for that
submission: the same resizing and interpolation, the same normalisation, the same conversion to 8 bit (rounding or truncation: copy your own, the
template's `run.py` uses the organizer baseline's), the same output sizes, the same device type if you can (the reference is whatever you used for the leaderboard;
say in `METHOD.md` if it was a CPU or an unusual GPU). Run your own script and `run.py` on the same images and check the outputs are identical.

## How a re-run is judged

We re-run on an NVIDIA GPU of the RTX 3090 class, with PyTorch 2.4.0 (the base image), and compare image by image with your leaderboard outputs and
metric by metric with your leaderboard scores. Tiny differences that come from the GPU or from rounding are tolerated within a tolerance we set from
measured run-to-run noise; anything larger is reviewed by a person before any decision. The tolerance is not published as a number.

## Check your repository before you hand in

```bash
python repro_check.py .                                        # layout, repro.yaml, weights link, signatures
python repro_check.py . --run --input /path/to/degraded --n 5  # also runs main() on 5 images and checks the outputs
python repro_check.py . --export-onnx                          # Track 2: also exports and checks the ONNX file
python repro_check.py . --download-weights                     # link-based repo: download the weights into weights/ first
```
No installation needed. For the closest match to our environment run it in the base image with the network off:

```bash
docker run --rm --network none --gpus all -v "$PWD":/pkg:ro -v /path/to/images:/in:ro -v /tmp/out:/out \
    pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime python /pkg/repro_check.py /pkg --run --input /in --output /out
```
(The base image is that PyTorch image plus numpy, pillow, opencv-python-headless, timm, einops, onnx, pyyaml, scipy,
scikit-image. Your `requirements.txt` is installed on top before the run.) The checker validates the *form* of your hand-in; whether
the results reproduce is decided by the organizers by running it.

## What happens next

From October 28 we fetch each repository at the commit you name, build the environment, run `run.py` with no network, score
the outputs and compare with your reported testing-phase result. If something fails for a packaging reason we write to the
address in `repro.yaml`. Anything that is not a clean match is reviewed by a person; no team is disqualified automatically.
Final rankings are announced by October 30, 2026 (AoE).
