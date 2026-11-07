# BRIDGE-ISL

**Goal:** watch someone sign in Indian Sign Language (ISL) and write the English sentence.

We don't use the video pixels. We use **pose data**: the positions of the body, hands, and face in every frame (like a stick figure). The model learns:

```
stick-figure movement  ──►  English sentence
```

This is a class project and research paper. It uses the **iSign** dataset, which is for **research only**.

---

## The big picture

| # | Step | What it does (simply) | Who | Status |
|---|------|------------------------|-----|--------|
| 1 | **Download** | Get the stick-figure files for ~30,000 sign clips | owner | ✅ done |
| 2 | **Clean up** | Make every clip the same format so the model can read it | owner | ✅ done |
| 3 | **Train the encoder** (Stage 0) | Teach the model to *understand movement* | owner | ✅ done |
| 4 | **Hand-off** | Package the encoder's results (~0.75 GB) for a teammate | owner | ✅ done |
| 5 | **Connect to a language model** (Stage 1) | Teach it to turn movement into *words* | teammate | ⬜ next |
| 6 | **Fine-tune** (Stage 2) | Make the sentences better | teammate | ⬜ |
| 7 | **Final polish + test score** | Train everything together, measure once on test | owner | ⬜ |

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

## 4. Hand-off to a teammate (owner)

**Why not just send the `.pt` file?** The `.pt` file is the trained *model*. To train the next part (the MLP), your teammate also needs *examples*: "this signing → this sentence". Those examples come from the 66 GB of data, which is too big to send.

**The trick:** we run the encoder over every clip **once** and save what it outputs. That is small (~0.75 GB), and it is exactly what the MLP needs as input. Your teammate does not need the raw data at all.

```bash
uv run python -m src.export --zip        # takes ~2 min → creates handoff/ and handoff.zip
```

**What's inside `handoff/`:**

| File | What it is |
|------|------------|
| `embeddings_train.npy` (+ `_val`, `_test`) | the encoder's output for every clip (256 numbers per step) |
| `index_train.csv` (+ `_val`, `_test`) | which rows belong to which clip, plus the **English sentence** |
| `encoder_stage0.pt` | the trained encoder itself (needed later for the final step and demos) |
| `stats.npz`, `feature_layout.json` | needed to prepare **new** videos the same way |
| `info.json` | sizes, settings, scores |

**How to send it:**

1. Push the code: `git add -A && git commit -m "..." && git push`
2. Share `handoff.zip` **privately** (private Google Drive link or a private Hugging Face repo). Never public.
3. Your teammate should also accept the [iSign terms](https://huggingface.co/datasets/Exploration-Lab/iSign) on Hugging Face (1 minute, no download).

---

## 5. Teammate: build the MLP and train (Stage 1 + Stage 2)

```
encoder output ──► PROJECTOR (the MLP) ──► Flan-T5 (language model) ──► English sentence
 (in handoff/)       you build this         already knows English
```

**Think of it like this:** the encoder speaks "sign language numbers", Flan-T5 speaks English. The projector is the **translator in the middle**.

### Step 1: Setup (10 min)

```bash
git clone <repo-url> && cd bridge
uv sync
uv add transformers peft sentencepiece sacrebleu
# unzip handoff.zip so the files are in bridge/handoff/
```

### Step 2: Check the data loads (1 min)

```bash
uv run python -c "from src.data import EmbeddingDataset; ds=EmbeddingDataset('handoff','train'); print(len(ds), ds[0]['h'].shape, ds[0]['text'])"
```

You should see `25030`, a shape like `[40, 256]`, and an English sentence.

### Step 3: Build the projector (`src/models/projector.py`)

A small MLP: 256 numbers in, 768 numbers out (the size Flan-T5-base expects).

```python
import torch.nn as nn

class Projector(nn.Module):
    def __init__(self, in_dim=256, out_dim=768, hidden=768, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim))

    def forward(self, h):          # (batch, steps, 256) -> (batch, steps, 768)
        return self.net(h)
```

### Step 4: Stage 1: train only the projector (`src/training/stage1.py`)

Flan-T5 is **frozen** (it doesn't change). Only the projector learns to "talk" to it.

The core of the training loop:

```python
from src.data import EmbeddingDataset, collate_embeddings
# load Flan-T5:  t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base"); t5.requires_grad_(False)

for batch in loader:                                   # batch.h, batch.mask, batch.texts
    labels = tokenizer(batch.texts, padding=True, return_tensors="pt").input_ids
    labels[labels == tokenizer.pad_token_id] = -100    # ignore padding
    loss = t5(inputs_embeds=projector(batch.h),
              attention_mask=batch.mask.long(), labels=labels).loss
    loss.backward(); optimizer.step(); optimizer.zero_grad()
```

- Settings: learning rate `3e-4`, batch size `32`, ~10 epochs.
- After each epoch, generate sentences on **val** and score them:

```python
ids = t5.generate(inputs_embeds=projector(batch.h), attention_mask=batch.mask.long(),
                  num_beams=5, max_new_tokens=64)
sentences = tokenizer.batch_decode(ids, skip_special_tokens=True)
# score with sacrebleu.corpus_bleu(sentences, [references])
```

- **Expect:** at first, generic sentences. Then more correct topics and words. BLEU-4 around **1–4** is normal here.

### Step 5: Stage 2: add LoRA (`src/training/stage2.py`)

Now Flan-T5 is allowed to adapt **a little**, through small add-on weights called LoRA (~1–2% of its size). Its main weights stay frozen.

```python
from peft import LoraConfig, get_peft_model
projector.load_state_dict(torch.load("checkpoints/stage1/projector.pt"))   # start from Stage 1
t5 = get_peft_model(t5, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                   target_modules=["q", "k", "v", "o"]))
# same loop as Stage 1, ~10 epochs. Learning rates: projector 1e-4, LoRA 2e-4
t5.save_pretrained("checkpoints/stage2/lora")
torch.save(projector.state_dict(), "checkpoints/stage2/projector.pt")
```

- **Expect:** better word choice. BLEU-4 around **3–7** is a good result for this data.

### Step 6: Send back to the owner

| What | How |
|------|-----|
| code (`projector.py`, `stage1.py`, `stage2.py`) | git push |
| `checkpoints/stage2/projector.pt` + `checkpoints/stage2/lora/` | private Drive (not git) |
| val scores per epoch + ~20 examples (real vs generated sentence) | for the paper |

### Rules

1. **Never tune on `test`.** Use `val`. The test score is measured **once**, at the very end.
2. Use `google/flan-t5-base` (or `flan-t5-small` for quick tries).
3. Write down the settings of every run (for the paper's comparison table).
4. Don't share the hand-off files outside the team.

**Time:** Stage 1 ~2–3 h, Stage 2 ~3–5 h on a laptop with a GPU. On a laptop **without** a GPU it is much slower, so use a free GPU on Google Colab or Kaggle (upload `handoff.zip` there), or use `flan-t5-small` / `EmbeddingDataset(..., max_clips=2000)` for quick tests.

---

## 6. Final step (owner)

When the teammate sends back the projector + LoRA: load encoder + projector + LoRA, let the **encoder** learn a little too (low learning rate, 1–2 epochs on the real data), then measure **once** on the test split. That number goes in the paper.

---

## Project layout

```
bridge/
├── src/
│   ├── config.py          settings and paths
│   ├── env.py             reads HF_TOKEN from .env
│   ├── extraction/        step 1: download
│   ├── preprocessing/     step 2: clean up, split, stats
│   ├── data/              loads clean clips (and hand-off embeddings) for training
│   ├── models/            the encoder (+ helper heads)
│   ├── training/          step 3: Stage 0 training
│   └── export/            step 4: build the hand-off package
├── data/                  (not in git) all data
├── checkpoints/           (not in git) trained models
├── handoff/               (not in git) the package for the teammate
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
uv run python -m src.export --zip                              # 4. hand-off package
```

These work the same on macOS, Windows, and Linux. For long runs, keep the computer awake (see step 3).

Every command has `--help` to list its options, e.g. `uv run python -m src.training.stage0 --help`.

## Citation

Dataset: Joshi et al., *iSign: A Benchmark for Indian Sign Language Processing*, Findings of ACL 2024.
