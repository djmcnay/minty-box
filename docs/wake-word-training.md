# Training a Custom Wake Word Model for openWakeWord

How to create a custom "Araminta" wake word model for use with the Minty Box
wake word listener. Training uses 100% synthetic speech — no manual audio
recording required. You provide the wake word and the tooling handles the
rest.

## Why "Araminta"

Four syllables, phonetically distinctive, vanishingly unlikely to appear in
normal conversation. When talking *about* Araminta you'll use "Minty", so
there's no collision between the wake word and casual reference. This is
the ideal wake word profile: long enough for good discrimination, rare
enough for very low false activations.

## Prerequisites

- A GitHub or email account (for sign-in)
- ~45–60 minutes (mostly unattended — training runs on cloud GPUs)
- No local setup required

## Step 1: Go to openWakeWord.com

The training service is at:

**https://openwakeword.com/**

Click **"Sign In"** in the top-right corner. You can sign in with GitHub or
email — no subscription required. The service is built and maintained by
David Scripka, the author of openWakeWord.

## Step 2: Enter Your Wake Word

Type "Araminta" on the training page. The service accepts any word or phrase
in any language.

**Phonetic spelling tip:** If you're unsure how the TTS voices will pronounce
"Araminta", you can use the phonetic spelling `aramintha` to nudge them
toward the right sound. The model name will still be `araminta` — the
phonetic version is only used during synthetic speech generation.

## Step 3: Preview with Multiple AI Voices

Before committing to training, the service lets you **preview** how
"Araminta" sounds when spoken by different AI voices. Listen to a few. If
the pronunciation sounds wrong — "air-a-min-tuh" when you want
"ah-ra-min-tah" — adjust the phonetic spelling and preview again. This is
the single biggest advantage over the Colab workflow: you catch TTS
pronunciation issues *before* training.

## Step 4: Set Training Parameters (or Skip)

The service offers a few configuration options:

| Parameter | Recommendation |
|-----------|---------------|
| Sample count | Default (~5,000) is fine for a first model |
| Language mix | English-only for "Araminta" |
| Training intensity | Default |

You can leave everything at defaults and hit train — the defaults produce
a solid model. If you later want lower false activations, retrain with
higher sample counts or a custom verifier model.

## Step 5: Train

Click **"Train"** (or equivalent). The model is trained on cloud GPUs and
typically finishes in under an hour. You'll get a notification when it's
done.

## Step 6: Download the Model

When training completes, download the `.onnx` file:

```
araminta.onnx
```

This is a standard openWakeWord ONNX model — identical in format to the
ones that ship with the `openwakeword` Python package. It's typically
sub-100 KB and runs entirely on-device.

## Step 7: Install on the Pi

Copy the `.onnx` file to the Minty Box:

```bash
# From your Mac (adjust source path)
scp ~/Downloads/araminta.onnx pi@minty-box.local:~/minty-box/models/

# Or place it directly in the minty-box repo
cp araminta.onnx ~/Documents/GitHub/minty-box/models/
```

## Step 8: Use with the Listener

```bash
cd ~/Documents/GitHub/minty-box
uv run python -m minty_box.wake --model models/araminta.onnx
```

Or from Python:

```python
from minty_box.wake import WakeWordListener

def on_wake(word: str, score: float) -> None:
    print(f"Heard you: {word} ({score:.3f})")
    # Trigger STT pipeline here

listener = WakeWordListener(
    on_wake=on_wake,
    model_paths=["models/araminta.onnx"],
)
listener.start()  # blocks
```

## Performance Expectations

openWakeWord's baseline targets (what the models are trained to achieve):

| Metric | Target |
|--------|--------|
| False reject rate | < 5% |
| False activations | < 0.5 per hour |
| Latency | ~80ms (single frame) |

Custom models typically match or exceed these. Four-syllable wake words
like "Araminta" tend to have even lower false activation rates than the
built-in two-syllable models (like "Alexa").

## Tuning

If false activations are too high:

1. **Raise the threshold.** Try 0.6 or 0.7 instead of the default 0.5.
   ```bash
   uv run python -m minty_box.wake --model models/araminta.onnx --threshold 0.7
   ```

2. **Enable VAD gating** (on by default). This filters predictions against
   Silero VAD so non-speech noise (kettle, woodburner, dog) can't trigger
   the wake word.

3. **Train a verifier model.** openWakeWord supports a second-stage filter
   trained specifically on your voice. This dramatically reduces false
   activations from other speakers saying "Araminta". See
   [openWakeWord custom verifier docs](https://github.com/dscripka/openWakeWord/blob/main/docs/custom_verifier_models.md).

If the wake word isn't detected reliably:

1. **Lower the threshold.** Try 0.3 or 0.4.
2. **Check microphone input.** Run the microphone test from
   [respeaker-lite-setup.md](respeaker-lite-setup.md).
3. **Retrain with more variation.** Increase the sample count or train
   with more voices at openWakeWord.com.

## Advanced: Multi-Wake-Word

You can run multiple wake words simultaneously. For example, "Araminta"
as the primary wake word and "hey jarvis" as a secondary:

```bash
uv run python -m minty_box.wake --model models/araminta.onnx --model alexa_v0.1.onnx
```

The `on_wake` callback receives the model name, so you can route different
wake words to different actions.
