# Speakinput Docs

Reference documentation for the speakinput internals. For install, usage,
troubleshooting, and the user-facing config reference, see the
[root README](../README.md). For the full design document (architecture,
error recovery, pluggable STT engines, microphone failover, injector fallback
chain, listener self-healing, and the complete configuration reference), see
[DESIGN.md](../DESIGN.md).

## Index

- [architecture.md](architecture.md) — system overview, module map, threading
  model, lock discipline, press lifecycle, layered prompt construction, model
  sharing, and platform backends.
- [error-recovery.md](error-recovery.md) — the layered recovery system:
  listener death, CGEventTap disable, system sleep, dead audio stream, audio
  device gone, CoreAudio HAL deadlock, subprocess hangs, Ctrl-C/shutdown, and
  transcription/injection failure. Includes shutdown order and test coverage.
- [platforms.md](platforms.md) — macOS, Linux, and Windows support: backend
  selection for hotkey detection and text injection, permissions, clipboard
  handling, and the cross-platform differences table.
- [research/text-injection-unicode.md](research/text-injection-unicode.md) —
  what each injection backend can type (wtype, ydotool, pynput on macOS, X11,
  Windows) and what Wayland terminals do with a Unicode keysym.
- [research/gnome-wayland-injection.md](research/gnome-wayland-injection.md) —
  how GNOME/Mutter exposes input injection (EIS over the RemoteDesktop
  portal, the on-screen-keyboard path, IBus) and why Unicode still cannot be
  injected there.
