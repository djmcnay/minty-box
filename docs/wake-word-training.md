# Training a Custom Wake Word Model for openWakeWord

How to create a custom "Araminta" wake word model for use with the Minty Box
wake word listener. Training uses 100% synthetic speech — no manual audio
recording required. You provide the two words ("Araminta") and the tooling
handles the rest.

## Why "Araminta"

Four syllables, phonetically distinctive, vanishingly unlikely to appear in
normal conversation. When talking *about* Araminta you'll use "Minty", so
there's no collision between the wake word and casual reference. This is
the ideal wake word profile: long enough for good discrimination, rare
enough for very low false activations.

## Prerequisites

- A Google account (for Colab)
- 45–60 minutes (mostly unattended)
- No GPU required — training runs on Colab's free tier

## Step 1: Open the Training Notebook

openWakeWord provides a one-click Colab notebook for this exact workflow:

**https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb**

Click "Copy to Drive" to get your own editable copy.

## Step 2: Configure the Wake Word

In the notebook's configuration cell, set:

```python
target_word = "aramintha"  # phonetic spelling works better for TTS
# The model name will be "araminta" — cleaner
model_name = "araminta"
```

Use `aramintha` for the TTS prompt (phonetic) but `araminta` as the model
file name. The notebook generates thousands of synthetic clips with varied
voices, accents, room acoustics, and background noise saying "Araminta".

## Step 3: Configure Negative Examples

The notebook needs audio that is NOT the wake word for training. Accept the
defaults — it uses the FSDD50K and MUSAN datasets (speech + noise) which
provide excellent coverage. No need to change anything here.

## Step 4: Run All Cells

Runtime → Run all. This will:

1. Generate ~5,000 synthetic "Araminta" clips from multiple TTS voices
2. Mix them with background noise and room impulse responses
3. Extract the speech embeddings (Google's pre-trained model)
4. Train a lightweight classifier on top
5. Validate against held-out test data
6. Run a false-accept test against 5+ hours of conversational speech
7. Output the model as `araminta.onnx`

The training runs on Colab's free CPU tier and finishes in ~45 minutes.

## Step 5: Download the Model

When training completes, the notebook saves the model to the Colab
runtime. Download it:

```python
from google.colab import files
files.download('araminta.onnx')
```

## Step 6: Install on the Pi

Copy the `.onnx` file to the Minty Box:

```bash
# From your Mac (adjust source path)
scp ~/Downloads/araminta.onnx pi@minty-box.local:~/minty-box/models/

# Or place it directly in the minty-box repo
cp araminta.onnx ~/Documents/GitHub/minty-box/models/
```

## Step 7: Use with the Listener

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
3. **Retrain with more variation.** The default notebook uses 5 voices;
   the advanced notebook supports up to 20.

## Advanced: Multi-Wake-Word

You can run multiple wake words simultaneously. For example, "Araminta"
as the primary wake word and "hey jarvis" as a secondary:

```bash
uv run python -m minty_box.wake --model models/araminta.onnx --model alexa_v0.1.onnx
```

The `on_wake` callback receives the model name, so you can route different
wake words to different actions.
