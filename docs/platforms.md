# Platform Guide

Speakinput supports macOS (primary) and Linux (secondary). Platform-specific code is isolated in each module via backend selection.

## macOS

### Requirements
- Python 3.11+
- Input Monitoring permission (System Settings → Privacy & Security → Input Monitoring)
- Accessibility permission (System Settings → Privacy & Security → Accessibility)
- PortAudio (installed via `sounddevice`)
- Optional: `nowplaying-cli` (`brew install nowplaying-cli`) for media pause/resume across all apps (Safari, Firefox, Music, Spotify, etc.). Without it, only Spotify and Music.app are paused via osascript.

### Defaults
- Hotkey: `alt_r` (Right Option)
- Injection: pynput (Cmd+V for Unicode)
- Media: nowplaying-cli (all apps via system Now Playing), osascript fallback (Spotify/Music.app only)
- Menu bar: rumps (NSStatusBar)

### macOS-Specific Behavior

**CGEventTap Self-Healing (`_mac_tap_heal.py`):**
macOS silently disables CGEventTaps on sleep/wake, focus transitions, and permission changes. The pynput thread stays alive but delivers no events. The monkey-patch in `_mac_tap_heal.py` calls `CGEventTapIsEnabled()` every 1s and re-enables the tap if disabled. Imported at module load by `hotkey.py` before any listener is constructed.

**CoreAudio HAL Deadlock:**
`stream.stop()` / `stream.close()` can wedge indefinitely due to HAL mutex contention (especially after sleep/wake or USB/Bluetooth mic glitch). `AudioRecorder.close()` bounds this with a 1.5s helper thread timeout.

**Sleep/Wake Recovery:**
`_on_system_sleep()` closes the audio recorder (the stream goes silent after wake) and aborts any orphaned press. It does NOT restart the listeners — the old listener's cleanup races with a new listener's tap enable, so the app relies on the `_mac_tap_heal.py` self-healing mechanism to re-enable the CGEventTap within ~1s. If self-healing fails, the heartbeat backstop (30s stale threshold) detects the dead listener and restarts it. Detection uses wall/monotonic clock skew > 5s.

### Permissions Notes
- Both Input Monitoring AND Accessibility are needed for pynput global hotkeys
- The first press after granting permissions may require a speakinput restart
- Bluetooth/USB mic changes may trigger device fallback with a warning

## Linux

### Requirements
- Python 3.11+
- For evdev (recommended): user must be in `input` group
- For wtype: wlroots-based compositor (Sway, Hyprland, etc.)
- For ydotool: `ydotoold` daemon running
- PortAudio (ALSA or PulseAudio)

### Defaults
- Hotkey: `ctrl_r` (Right Ctrl)
- Injection: wtype → ydotool → pynput (fallback chain)
- Media: playerctl (MPRIS D-Bus)
- Menu bar: stderr

### Hotkey Backend Selection

```
if sys.platform == "linux":
    probe_evdev()  # scan /dev/input/event*
    if keyboard found:
        use evdev  # works on Wayland, X11, headless
    else:
        fall back to pynput  # X11 only
else:
    use pynput  # macOS / Windows
```

**evdev** reads `/dev/input/event*` directly — works on Wayland, X11, and headless. One thread per keyboard device. Shared latch flag deduplicates across devices.

**pynput** on Linux uses XRecord — requires X11. Does not work on pure Wayland.

### Injection Backend Selection

```
if macOS/Windows:
    TypingInjector (pynput)
elif Linux Wayland:
    wtype (preferred) → ydotool (fallback) → pynput (last resort)
elif Linux X11:
    TypingInjector (pynput)
```

**wtype** uses the wlroots virtual-keyboard protocol. It uploads its own XKB keymap with one keycode per distinct character, so ASCII and Unicode (CJK, accents, emoji) are both typed directly. No clipboard, and the text therefore lands in terminal emulators, which paste on `Ctrl+Shift+V` and ignore `Ctrl+V`. wtype needs a short pause after the keymap upload before the first key press (the compositor and client apply the new keymap asynchronously; without it the first character is dropped) — that is `_WTYPE_KEYMAP_SETTLE_MS` in `injector.py`.

**ydotool** uses uinput. Needs `ydotoold` daemon. Its `type` command is an ASCII-only keycode table, so non-ASCII text goes through the clipboard and `Ctrl+V`. That works in GUIs but not in terminals.

**GNOME / KDE** implement neither the virtual-keyboard protocol nor a way to inject Unicode: Mutter's EIS seat (since 45) exposes keyboard/pointer devices with keycodes only, and `zwp_input_method_v1` is not implemented. On those desktops the only routes that carry CJK are an IBus engine or a compositor extension. See `docs/research/gnome-wayland-injection.md`.

### Clipboard on Linux
Uses `pyperclip` which falls back to `wl-paste`/`wl-copy` on Wayland or `xclip`/`xsel` on X11.

## Windows (Partial/Untested)

- Hotkey: pynput (Win32 hooks)
- Injection: pynput (SendInput)
- Media: PowerShell SMTC
- Default key: `ctrl_r`

## Cross-Platform Differences

| Feature | macOS | Linux |
|---------|-------|-------|
| Hotkey backend | pynput only | evdev (preferred) → pynput |
| CGEventTap healing | Yes (`_mac_tap_heal.py`) | N/A |
| Sleep recovery | Recorder close + tap self-heal (no listener restart) | Same (clock skew detection) |
| Unicode injection | pbcopy → Cmd+V | wtype types it directly; ydotool/pynput use wl-copy → Ctrl+V |
| Media control | nowplaying-cli → osascript fallback | playerctl |
| Menu bar | rumps (NSStatusBar) | stderr |
| Permissions | Input Monitoring + Accessibility | `input` group for evdev |
