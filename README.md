# BRIDGE-ISL

**Goal:** watch someone sign in Indian Sign Language (ISL) and write the English sentence.

We don't use the video pixels. We use **pose data**: the positions of the body, hands, and face in every frame (like a stick figure). The model learns:

```
stick-figure movement  ──►  English sentence
```

This is a class project and research paper. It uses the **iSign** dataset, which is for **research only**.

---

## The big picture

There are 5 steps. Steps 1–3 are done.

| # | Step | What it does (simply) | Status |
|---|------|------------------------|--------|
| 1 | **Download** | Get the stick-figure files for ~30,000 sign clips | ✅ done |
| 2 | **Clean up** | Make every clip the same format so the model can read it | ✅ done |
| 3 | **Train the encoder** (Stage 0) | Teach the model to *understand movement* | ✅ done |
| 4 | **Connect to a language model** (Stage 1) | Teach it to turn movement into *words* | ⬜ next |
| 5 | **Fine-tune** (Stage 2) | Make the sentences better | ⬜ later |

---

## 0. One-time setup

You need [uv](https://docs.astral.sh/uv/) (a Python tool).

```bash
cd bridge
uv sync                      # installs everything (same versions for everyone)
```

**Get access to the data:**

1. Make a free account on [Hugging Face](https://huggingface.co).
2. Open the [iSign dataset page](https://huggingface.co/datasets/Exploration-Lab/iSign) and accept the terms.
3. Create a token: Hugging Face → Settings → Access Tokens → *New token* (type: read).
4. Put it in a file called `.env`:

```bash
cp .env.example .env          # macOS / Linux
copy .env.example .env        # Windows
# open .env and paste your token:
# HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

> ⚠️ Never share `.env` or the `data/` folder. The dataset license does not allow it.

---

## 1. Download the data

**What happens:** the dataset is one giant 170 GB zip file online. Instead of downloading all of it, we pull out only the clips we need, one by one.

```bash
uv run python -m src.extraction --limit 30000
```

- `--limit 30000` → about 30,000 clips (~55 GB). Leave it out to get all ~127,000 clips (~170 GB).
- It takes a few hours. **If it stops, run the same command again.** It skips what is already downloaded.
- Result: `data/poses/<clip>.pose` (one file per clip) and `data/iSign_v1.1.csv` (the English sentence for each clip).

---

## 2. Clean up (preprocess)

**What happens, step by step:**

1. **Pick the important points:** body (33), left hand (21), right hand (21), and 44 face points (lips, eyebrows, eyes).
2. **Fill gaps:** if a hand was not detected for a few frames, guess where it was from the frames around it.
3. **Same speed for all:** some videos are 25 fps, some 29 fps. Make all of them 25 fps.
4. **Same size for all:** move the "center" to between the shoulders and measure everything in shoulder-widths. Now it doesn't matter how close the person stood to the camera.
5. **Throw away bad clips:** no sentence, almost no hands visible, or too short.
6. **Split the data:**
   - **train (80%)**: the model learns from these
   - **val (10%)**: we check progress on these while training
   - **test (10%)**: used **once at the very end** for the final score

   All clips from the same video always stay in the same group, so the model can't "cheat".
7. **Measure averages** (from train only) so all numbers can be put on the same scale later.

```bash
uv run python -m src.preprocessing
```

- Takes ~30 minutes. If it stops, run it again. It skips finished clips.
- Quick test on 200 clips: `uv run python -m src.preprocessing --max-clips 200 --out-dir test_outputs/pp`

**What you get (in `data/`):**

| File | What it is |
|------|------------|
| `processed/<clip>.npy` | the cleaned clip: a table of *frames × 492 numbers* |
| `metadata.csv` | one row per clip: id, video, split (train/val/test), sentence, length |
| `dropped.csv` | clips we threw away and why |
| `stats.npz` | average and spread of every number (from train only) |
| `feature_layout.json` | which of the 492 numbers is body / hands / face |
| `splits/*.txt` | which videos are in train / val / test |

**Peek at the data:**

```bash
uv run python -c "import numpy as np; a=np.load('data/processed/-02o_0vVwzI--0.npy'); print(a.shape)"
uv run python -c "import numpy as np; s=np.load('data/stats.npz'); print(s['mean'][:6], s['std'][:6])"
```

---

## 3. Train the encoder (Stage 0)

**What is the encoder?** The part of the model that *watches the movement* and turns it into a list of numbers that describe "what is happening".

**How it learns (two games):**

1. **Fill in the blank:** we hide ~40% of the frames and it must guess the hidden movement. This teaches it how signing moves.
2. **Guess the words:** it guesses which words are in the sentence. This teaches it that movements have *meaning*.

```bash
uv run python -m src.training.stage0
```

**Keep the computer awake while it runs** (if it sleeps, training pauses):

| OS | How |
|----|-----|
| **macOS** | put `caffeinate -i` in front: `caffeinate -i uv run python -m src.training.stage0` |
| **Windows** | Settings → System → Power → *When plugged in, put my device to sleep after* → **Never**. Or run `powercfg /change standby-timeout-ac 0` before, and `powercfg /change standby-timeout-ac 30` after to restore it. |
| **Linux** | put `systemd-inhibit` in front: `systemd-inhibit uv run python -m src.training.stage0` |

Keep the laptop plugged in. The same applies to the long download in step 1.

- ~2 minutes per epoch on an M5 MacBook (other laptops may be slower or faster). It **stops by itself** when it stops improving (~45–60 min total).
- **If it stops early** (laptop closed, etc.), continue from where it stopped:

```bash
uv run python -m src.training.stage0 --resume checkpoints/stage0/last.pt
```

**Check that the code works first (1 minute):**

```bash
uv run python -m src.training.stage0 --overfit --epochs 40 --num-workers 0 --out-dir test_outputs/s0
```

The loss should go down. If it does, everything is wired correctly.

**Reading the progress line:**

```
epoch  7 | train 0.99 (recon 0.35, word 6.43) | val 1.00 (recon 0.32, word R@10 0.112) | lr 2.7e-04 | 2.3 min
```

| Part | Meaning | Good when… |
|------|---------|------------|
| `recon` | how wrong the "fill in the blank" guesses are | it goes **down** |
| `word` | how wrong the word guesses are | it goes **down** |
| `R@10` | how many of the real words are in its top-10 guesses | it goes **up** (must beat 0.067) |
| `val` | same scores on clips it never trained on | **this is the one that matters** |

**What you get (in `checkpoints/stage0/`):**

| File | What it is |
|------|------------|
| `best.pt` | **the trained encoder**. This is what Stage 1 uses |
| `last.pt` | the latest state, used for `--resume` |
| `log.csv` | scores for every epoch |
| `config.json` | the exact settings used |

**Our result:** best at epoch 17, recon **0.295** (just guessing the average scores 0.80) and R@10 **0.115** (just guessing common words scores 0.067).

**Useful options:**

```bash
--epochs 30              # max number of rounds
--batch-size 32          # clips per step
--lr 3e-4                # learning speed
--word-weight 0          # turn off the "guess the words" game
--feature-groups body left_hand right_hand    # no face (to compare with/without face)
--no-augment             # turn off random rotation/speed changes
```

**Use the trained encoder in code:**

```python
from src.models import PoseEncoder
encoder = PoseEncoder.from_checkpoint("checkpoints/stage0/best.pt")
# input:  (batch, frames, 360) → output: (batch, frames/4, 256)
```

---

## 4. Next: connect to a language model (Stage 1). Coming soon

```
movement ──► ENCODER ──► PROJECTOR (small MLP) ──► Flan-T5 (language model) ──► sentence
             (done)       (to build)                 (frozen, already knows English)
```

The projector translates the encoder's numbers into the language model's "language". Only the encoder and projector learn; Flan-T5 stays as it is.

---

## Project layout

```
bridge/
├── src/
│   ├── config.py          settings and paths
│   ├── env.py             reads HF_TOKEN from .env
│   ├── extraction/        step 1: download
│   ├── preprocessing/     step 2: clean up, split, stats
│   ├── data/              loads clean clips for training
│   ├── models/            the encoder (+ helper heads)
│   └── training/          step 3: Stage 0 training
├── data/                  (not in git) all data
├── checkpoints/           (not in git) trained models
├── pyproject.toml, uv.lock
└── .env.example           copy to .env and add your token
```

## All commands in one place

```bash
uv sync                                                        # setup
uv run python -m src.extraction --limit 30000                  # 1. download
uv run python -m src.preprocessing                             # 2. clean up
uv run python -m src.training.stage0                           # 3. train encoder
uv run python -m src.training.stage0 --resume checkpoints/stage0/last.pt   # 3. continue
```

These work the same on macOS, Windows, and Linux. For long runs, keep the computer awake (see step 3).

Every command has `--help` to list its options, e.g. `uv run python -m src.training.stage0 --help`.

## Citation

Dataset: Joshi et al., *iSign: A Benchmark for Indian Sign Language Processing*, Findings of ACL 2024.
