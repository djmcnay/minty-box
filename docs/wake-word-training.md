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

---

## Two Training Options

You can train a custom model using either of these approaches. Both produce
the same `.onnx` file — identical in format to the models that ship with
the `openwakeword` Python package (sub-100 KB, runs entirely on-device).

| | openWakeWord.com | Google Colab |
|---|---|---|
| Cost | Paid (subscription required) | Free |
| Setup | Sign in with GitHub/email | Google account + copy a notebook |
| Training hardware | Cloud GPUs | Colab CPU (free tier) |
| Training time | ~30–45 min | ~45–60 min |
| Voice preview | ✅ Preview with multiple AI voices before training | ❌ No preview — train and see |
| Languages | 20+ languages | English (easily extended) |
| Ease | Click a button | Run cells in order |

---

## Option A: openWakeWord.com (Paid, Easier)

### Prerequisites

- A GitHub or email account
- Paid subscription (pricing on the site)
- ~30–45 minutes (mostly unattended)

### Steps

1. Go to **https://openwakeword.com/** and sign in.

2. Enter "Araminta" as the wake word. Any word or phrase in any language
   is accepted.

   **Phonetic spelling tip:** If you're unsure how the TTS voices will
   pronounce "Araminta", you can use the phonetic spelling `aramintha`
   to nudge them toward the right sound. The model name will still be
   `araminta` — the phonetic version is only used during synthetic
   speech generation.

3. **Preview** with multiple AI voices before training. This catches
   pronunciation issues early. If "air-a-min-tuh" isn't what you want
   (try "ah-ra-min-tah"), adjust the spelling and preview again.

4. Adjust training parameters or leave at defaults (~5,000 samples,
   English-only, default intensity).

5. Click **Train** and wait. Cloud GPU training typically finishes in
   under an hour. Download `araminta.onnx` when complete.

---

## Option B: Google Colab (Free, Slightly More Manual)

### Prerequisites

- A Google account (for Colab)
- ~45–60 minutes (mostly unattended)
- No GPU required

### Steps

1. Open the Colab notebook:

   **https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb**

   Click **"Copy to Drive"** to get your own editable copy.

2. In the notebook's configuration cell, set:

   ```python
   # The word the TTS engines will speak — phonetic spelling helps.
   target_word = "aramintha"

   # The output model name.
   model_name = "araminta"
   ```

   The notebook generates thousands of synthetic clips with varied
   voices, accents, room acoustics, and background noise.

3. Accept the defaults for negative examples (FSDD50K and MUSAN
   datasets — speech + noise). No changes needed.

4. Runtime → Run all. This will:
   - Generate ~5,000 synthetic "Araminta" clips from multiple TTS voices
   - Mix them with background noise and room impulse responses
   - Extract speech embeddings (Google's pre-trained model)
   - Train a lightweight classifier
   - Validate against held-out test data
   - Run a false-accept test against 5+ hours of conversational speech
   - Output the model as `araminta.onnx`

5. When training completes, download the model:

   ```python
   from google.colab import files
   files.download('araminta.onnx')
   ```

---

## Installing on the Pi

Whichever option you used, you now have an `araminta.onnx` file. Copy it
to the Minty Box:

```bash
# From your Mac (adjust source path)
scp ~/Downloads/araminta.onnx pi@minty-box.local:~/minty-box/models/

# Or place it directly in the minty-box repo
cp araminta.onnx ~/Documents/GitHub/minty-box/models/
```

---

## Use with the Listener

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

---

## Quick Testing with Built-in Models

While waiting for custom model training, you can test the listener with
the built-in wake words that ship with openWakeWord:

```bash
# Uses all built-in models: alexa, hey_mycroft, hey_jarvis, timer, weather
uv run python -m minty_box.wake
```

This is useful for verifying the audio pipeline works end-to-end before
committing to training a custom model.

---

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
   with more voices. The Colab advanced notebook supports up to 20 voices;
   openWakeWord.com offers similar controls.

## Advanced: Multi-Wake-Word

You can run multiple wake words simultaneously. For example, "Araminta"
as the primary and "hey jarvis" as a secondary:

```bash
uv run python -m minty_box.wake --model models/araminta.onnx --model alexa_v0.1.onnx
```

The `on_wake` callback receives the model name, so you can route different
wake words to different actions.
