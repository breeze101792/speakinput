# TODO

1. ~~we need this on setup vulkan, fix it on setup. "sudo pacman -S vulkan-headers"~~
   - Fixed: `vulkan-headers` is now included in the Arch Linux install step
   in `setup.sh` alongside `vulkan-icd-loader`.

## Future work

- **OpenVINO detection** — Intel's vendor stack needs a different probe
  than CUDA/Vulkan. The `setup.sh` script doesn't auto-probe for it yet.
- **Per-profile GPU device selection** — e.g. GPU 0 for English, GPU 1
  for Chinese, on a multi-GPU workstation.
- **StreamingTranscriber** — consume `AudioRecorder.chunk_generator()`
  for partial results while the key is still held.
- **CommandInjector** — interpret the transcription and dispatch to
  shell or an agent.
- **Streaming partial-results UI** — show text as you hold the key.
- **[hotkey].device_path** config knob for multi-keyboard setups where
  the user wants a specific evdev device.
- **Optional EVIOCGRAB mode** for users who want exclusive device grab.
- **Vosk engine** — lightweight offline engine for low-resource envs.
- **OpenAI Whisper API engine** — cloud-based, requires API key.