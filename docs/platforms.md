# Platform Guide

Speakinput supports macOS (primary) and Linux (secondary). Platform-specific code is isolated in each module via backend selection.

## macOS

### Requirements
- Python 3.11+
- Input Monitoring permission (System Settings → Privacy & Security → Input Monitoring)
- Accessibility permission (System Settings → Privacy & Security → Accessibility)
- PortAudio (installed via `sounddevice`)

### Defaults
- Hotkey: `alt_r` (Right Option)
- Injection: pynput (Cmd+V for Unicode)
- Media: osascript (Spotify, Music.app)
- Menu bar: rumps (NSStatusBar)

### macOS-Specific Behavior

**CGEventTap Self-Healing (`_mac_tap_heal.py`):**
macOS silently disables CGEventTaps on sleep/wake, focus transitions, and permission changes. The pynput thread stays alive but delivers no events. The monkey-patch in `_mac_tap_heal.py` calls `CGEventTapIsEnabled()` every 1s and re-enables the tap if disabled. Imported at module load by `hotkey.py` before any listener is constructed.

**CoreAudio HAL Deadlock:**
`stream.stop()` / `stream.close()` can wedge indefinitely due to HAL mutex contention (especially after sleep/wake or USB/Bluetooth mic glitch). `AudioRecorder.close()` bounds this with a 1.5s helper thread timeout.

**Sleep/Wake Recovery:**
`_on_system_sleep()` restarts all listeners (CGEventTaps are disabled) and closes the audio recorder (stream goes silent). Detection uses wall/monotonic clock skew > 5s.

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

**wtype** uses the wlroots virtual-keyboard protocol. ASCII: direct typing. Unicode: `wl-copy` + `Ctrl+V`.

**ydotool** uses uinput. Needs `ydotoold` daemon. Same clipboard approach for Unicode.

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
| Sleep recovery | Listener restart + recorder close | Same (clock skew detection) |
| Unicode injection | pbcopy → Cmd+V | wl-copy → Ctrl+V |
| Media control | osascript | playerctl |
| Menu bar | rumps (NSStatusBar) | stderr |
| Permissions | Input Monitoring + Accessibility | `input` group for evdev |
