# ReSpeaker Lite — Full Setup Guide

How to properly install a ReSpeaker Lite (XIAO ESP32S3) as a USB audio device
on a Raspberry Pi 5. Follow this *exactly*. Every pitfall documented here was
discovered the hard way on 8–9 May 2026.

**Hard requirement:** the ReSpeaker MUST work offline. No WiFi, no network
dependency — plug into the Pi and it appears as an ALSA sound card.

---

## Architecture

```
ReSpeaker mics → XMOS DSP → USB audio → Pi (ALSA hw:CARD=Lite,DEV=0)
    → faster-whisper (STT) → Hermes → Kokoro (TTS) → ReSpeaker speaker
```

---

## Hardware Overview

The ReSpeaker Lite has **two chips, two USB-C ports**:

| Chip | Role | USB-C port | Firmware needed |
|------|------|------------|-----------------|
| XMOS XU316 | Audio DSP — mic array, AEC, noise suppression, beamforming | XMOS side ("Seed Audio") | **USB audio firmware v2.0.7** |
| XIAO ESP32S3 | NOT USED in USB audio mode | XIAO side | N/A — keep as-is |

The board **ships with I2S firmware** on the XMOS. This must be replaced with
USB audio firmware for offline operation. If you skip this step, the Pi will
never see the device as a sound card.

### Speaker Connection

The mono enclosed speaker connects via the white JST 2-pin connector. The
onboard 5W amplifier drives it. The 3.5mm jack auto-detection circuit mutes the
JST speaker when a plug is inserted — leave the 3.5mm jack empty for JST speaker
output. No trick needed.

---

## Step 1: Prerequisites

```bash
sudo apt-get update
sudo apt-get install -y dfu-util
```

---

## Step 2: Identify the Board

Plug the ReSpeaker into the Pi using the **XMOS-side** USB-C port (labelled
"Seed Audio"). Use a **known-good data cable** — see the Cable Pitfalls section
below.

```bash
sudo dfu-util -l
```

You must see:
```
Found DFU: [2886:0019] ver=0200, devnum=..., cfg=1, intf=0, ...
name="ReSpeaker Lite"
alt=0, name="DFU FACTORY"
alt=1, name="DFU UPGRADE"
```

If you don't see this, check:
- Cable is a **data cable**, not charge-only
- Connected to the XMOS port, not the XIAO port
- `lsusb | grep 2886` shows the device at all

---

## Step 3: Flash USB Audio Firmware

Download and flash firmware v2.0.7:

```bash
curl -L -o /tmp/respeaker_lite_usb_dfu_firmware_v2.0.7.bin \
  https://raw.githubusercontent.com/respeaker/ReSpeaker_Lite/master/xmos_firmwares/respeaker_lite_usb_dfu_firmware_v2.0.7.bin

sudo dfu-util -R -e -a 1 -D /tmp/respeaker_lite_usb_dfu_firmware_v2.0.7.bin
```

The `-R` flag resets the board after flashing. Wait a few seconds for it to
re-enumerate.

---

## Step 4: Verify

```bash
cat /proc/asound/cards | grep "ReSpeaker Lite"
# Must show a card named "ReSpeaker Lite"
```

Also confirm firmware version:
```bash
sudo dfu-util -l
# Should show ver=0207 in the device descriptor
```

---

## Step 5: ALSA Device Naming

**Never use `hw:2,0`.** Card numbers change across reboots depending on USB
enumeration order. Always use the stable card-id form:

```
hw:CARD=Lite,DEV=0       # direct hardware device
plughw:CARD=Lite,DEV=0   # with automatic format/resampling conversion
```

Verified identity:
- `/proc/asound/cards`: `Lite` — USB-Audio, Seeed Studio, `2886:0019`

---

## Step 6: Power Configuration

The Pi 5 caps USB ports at 600mA unless `usb_max_current_enable=1` is set.
The ReSpeaker's combined XMOS + amp load triggers undervoltage without this.

### Two-part fix

**Part 1 — Placement matters.** `usb_max_current_enable=1` **MUST** be under
`[all]` in `/boot/firmware/config.txt`. If placed under `[cm5]` or any other
section that doesn't apply to the Pi 5, it silently does nothing.

Add to `/boot/firmware/config.txt`:
```
[all]
usb_max_current_enable=1
```

**Always verify it's actually active** — seeing it in config.txt is not enough:
```bash
sudo vcgencmd get_config usb_max_current_enable   # must return 1, NOT 0
```

**Part 2 — PSU matters.** The Pi 5 wants 5V/5A. Many 65W GaN chargers only
deliver 5V/3A (they use higher-voltage PD profiles for laptops). A true 5.1V/5A
PSU — the official Raspberry Pi 27W or equivalent — eliminates undervoltage
properly. **65W ≠ 5A at 5V.**

### Verification after reboot

```bash
sudo vcgencmd get_config usb_max_current_enable   # must be 1
sudo vcgencmd get_throttled                       # ideally 0x0
sudo vcgencmd pmic_read_adc                       # check EXT5V_V
```

### Interpreting throttled

`throttled` is a **latch** — once set, it stays set until cleared. It tells you
what HAS happened, not what IS happening.

| Bitmask | Meaning |
|---------|---------|
| 0x50000 | Historical: undervoltage + throttling have occurred |
| 0x50005 | Active: currently undervolted + throttled + historical |
| 0x0 | Clean |

---

## Step 7: PulseAudio Sink Volume

The ReSpeaker Lite in USB audio mode has **no hardware mixer**, so PulseAudio
manages a software volume on top.

**Critical pitfall:** after a cable swap or reboot, PulseAudio may silently
reset the sink volume to ~40% — even if you previously set it to 100%. This
wrecks the volume calibration.

### Check PulseAudio:

```bash
pactl list sinks | grep -A10 "ReSpeaker" | grep "Volume:"
# Must show: front-left: 65536 / 100% / 0.00 dB, front-right: 65536 / 100% / 0.00 dB
```

If it shows anything else (e.g., `26214 / 40%`), fix it:
```bash
pactl set-sink-volume \
  alsa_output.usb-Seeed_Studio_ReSpeaker_Lite_0000000001-00.analog-stereo 100%
```

---

## Volume Calibration

The ReSpeaker has no hardware mixer — volume is set via ffmpeg `volume=`
parameter (linear amplitude scaling). Human hearing is logarithmic, so the steps
feel uneven. This is correct, not a bug.

**These values are only valid when PulseAudio is at 100%.** If PulseAudio is
lower, all values are proportionally quieter.

| Perceived level | volume= | ~dB | When |
|----------------|---------|-----|------|
| 1/10 — whisper | 0.01 | -40 | Late night, won't travel upstairs |
| 2/10 — quiet night | 0.02 | -34 | Nighttime conversation |
| 3/10 — soft | 0.03 | -30 | |
| 5/10 — normal | 0.08 | -22 | Daytime |
| 7/10 — loud | 0.20 | -14 | Filling a room |
| 10/10 — max | 1.0 | 0 | Don't |

### Playback Commands

**Quick one-liner (ffplay)** — fastest for ad-hoc playback, no temp file:
```bash
ffplay -nodisp -autoexit -af "volume=0.08" <file.wav> 2>/dev/null
```

**Two-stage (ffmpeg → aplay)** — for format conversion or caching:
```bash
ffmpeg -v quiet -y -i <file> -af "volume=0.08" -ar 16000 -ac 2 -c:a pcm_s16le \
  /tmp/minty-play.wav \
  && aplay -D hw:CARD=Lite,DEV=0 /tmp/minty-play.wav
```

### Quiet Speaker Diagnosis

If the speaker seems too quiet, check in this order:

1. **PulseAudio sink volume** → fix to 100% if needed
2. **Physical connections** (JST to speaker, USB to Pi)
3. **ALSA card present:** `cat /proc/asound/cards | grep Lite`
4. **THEN** adjust ffmpeg `volume=` parameter

---

## Microphone Testing

The ReSpeaker presents as `hw:CARD=Lite,DEV=0` for capture (S16_LE, 2ch, 16kHz).

```bash
# 1. Record 5 seconds
arecord -D hw:CARD=Lite,DEV=0 -d 5 -f S16_LE -r 16000 -c 2 /tmp/mic_test.wav

# 2. Amplitude analysis
python3 -c "
import wave, struct, math
w = wave.open('/tmp/mic_test.wav')
n = w.getnframes()
data = w.readframes(n)
samples = struct.unpack(f'<{n*2}h', data)
peak = max(abs(s) for s in samples)
rms = math.sqrt(sum(s*s for s in samples)/len(samples))
print(f'Frames: {n}, Peak: {peak}, RMS: {rms:.1f}')
if peak < 100: print('⚠️  Very quiet — barely above noise floor')
elif peak < 1000: print('📢 Low but audible signal')
elif peak < 5000: print('📢 Decent signal')
elif peak < 15000: print('📢 Strong signal')
else: print('📢 Very loud / possibly clipping')
w.close()
"

# 3. Play back (amplify — recordings are quiet)
ffplay -nodisp -autoexit -af "volume=5.0" /tmp/mic_test.wav 2>/dev/null
```

**Verified working** 9 May 2026: peak 851, RMS 75.6 — low but audible signal at
normal speaking distance.

---

## USB Cable Pitfalls

A charge-only USB-A to USB-C cable will power the board (green LED lights up)
but the Pi will **never see it** in `lsusb` — no data lines. Always use a
**known data cable**.

### How to tell charge-only from data

Charge-only cables only wire VBUS + GND; they skip D+ and D- data lines. Rule
of thumb: if a cable came with a charger or power bank, assume charge-only until
proven otherwise.

### Cable Test Results (9 May 2026)

| Cable | Origin | Works? | Notes |
|-------|--------|--------|-------|
| Original | Nuphy Air keyboard | ✅ | Too long but data-capable |
| Cable 2 | Unknown drawer cable | ❌ | Charge-only — no lsusb, no dmesg connect |
| Cable 3 | Unknown, long | ✅ | Data-capable, too long |
| Cable 4 | Curiosity test | ✅ | Data-capable |

### Diagnostic Checklist

If the ReSpeaker goes missing after a cable swap:

```bash
# 1. Is it even on the USB bus?
lsusb | grep 2886

# 2. Did the kernel see it connect?
dmesg | tail -5 | grep -E "usb 1-1|2886"

# 3. Does ALSA know about it?
cat /proc/asound/cards | grep Lite

# 4. Full picture
lsusb && cat /proc/asound/cards && dmesg | tail -5
```

- dmesg shows disconnect but no connect → cable is charge-only or faulty
- dmesg shows connect but ALSA doesn't list it → wait 2 seconds (enumeration
  can lag); if still missing after 5 seconds, suspect firmware issue

### Transient Undervoltage

Disconnecting and reconnecting the ReSpeaker may trigger transient undervoltage
warnings (`Undervoltage detected!` / `Voltage normalised`). These are normal
during hot-plug events — the amp draws a spike on reconnection. Only investigate
if warnings persist after the device has settled.

---

## ESPHome / WiFi — NOT THE PATH

ESPHome WiFi mode was attempted and rejected on 8 May 2026. It requires WiFi
and cannot work offline or on different networks without reconfiguration. The
ESPHOME config and secrets are archived in `minty-box/esphome/` under
araminta-toolshed. **Do not pursue this path** without explicit approval.

---

## Known Mistakes (8–9 May 2026)

Recorded here so they are never repeated:

1. **Never verified `usb_max_current_enable` was active.** Saw it in config.txt
   and assumed done. `vcgencmd get_config` would have shown it returning `0`
   because it was under `[cm5]` instead of `[all]`.

2. **Pursued ESPHome/WiFi when the requirement was offline.** The USB audio
   firmware path was correct from the start.

3. **Misread DFU mode.** "DFU FACTORY" on the XMOS means no audio firmware —
   nothing to do with ESPHome flashing on the XIAO.

4. **Killed processes without tracing parent/child trees**, leaving zombie
   children. Always verify with `ps aux | grep` after killing.

5. **Blamed the PSU capacity** rather than checking whether the config change
   was actually being applied. The 65W GaN was fine — placement was wrong.

6. **Tried to play voice clips through Sonos.** Sonos is for music/Spotify/radio
   only. Voice clips, TTS output, and audio files go through the ReSpeaker Lite
   local speaker. Only route to Sonos if explicitly asked.

---

## Quick Sanity Check

After any reboot, cable swap, or "why isn't it working?" moment, run:

```bash
# Power
sudo vcgencmd get_config usb_max_current_enable   # 1
sudo vcgencmd get_throttled                       # 0x0

# USB
lsusb | grep 2886                                 # must show device

# Audio
cat /proc/asound/cards | grep Lite                # must show card

# PulseAudio
pactl list sinks | grep -A10 "ReSpeaker" | grep "Volume:"   # 100%
```
