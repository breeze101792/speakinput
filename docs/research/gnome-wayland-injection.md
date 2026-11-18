# Injecting keyboard input on GNOME Wayland (Mutter 51.1)

Research note. Read-only; no code changed.

## Question

On Arch Linux with Mutter 51.1 (GNOME 51), what mechanisms can a normal
application use to inject synthetic keyboard input into the focused client?
Specifically: Mutter's EIS/libei support and the portal handshake; existing
CLI/Python tools; GNOME Shell's virtual-input API; `dotool` versus `ydotool`;
and input-method paths (IBus, text-input-v3, input-method-v1).

## Local grounding

- Arch, `mutter 51.1-1`, `xdg-desktop-portal 1.22.1-2`,
  `xdg-desktop-portal-gnome 51.0-1`, `libei 1.6.0-2`,
  `ydotool 1.0.4-3`, `wtype 0.4-2`, `gtk3 1:3.24.52-2`,
  `vte3/vte4 0.84.1-1` (pacman, 2026-11-18).
- The running session is **not** GNOME: `DESKTOP_SESSION=swayfx-nvidia`,
  no `gnome-shell` process, `xdg-desktop-portal-gnome` inactive. The
  `org.freedesktop.portal.RemoteDesktop` interface was therefore not
  exported on this machine (`gdbus introspect`, 2026-11-18). Portal/EIS
  behaviour below comes from source and upstream docs, not from a live
  GNOME test here.
- Project backends: `config.example.toml:121-127`,
  `src/speakinput/injector.py:5-19`, `docs/platforms.md:391-395`.

## Findings

### 1. Mutter implements an EIS server. Since 45.0.

**Observed.** Mutter uses **libeis** (the server half of libei), not libei
itself: `src/backends/meta-eis.c` includes `<libeis.h>` and calls
`eis_new()` / `eis_backend_fd_add_client()`
(`https://gitlab.gnome.org/GNOME/mutter/-/blob/51.1/src/backends/meta-eis.c`,
lines 23, 320-323). The file exists in the `45.0` tag but not in `44.0`
(`/-/raw/45.0/src/backends/meta-eis.c` returns 383 lines; the 44.0 URL is
404). The 45.beta NEWS entry is "Support input capture and emulated input
with libei" (`NEWS`, line 1278 in 51.1).

**Observed.** The EIS seat advertises keyboard, pointer/button/scroll and
touch only — no `EIS_DEVICE_CAP_TEXT`
(`src/backends/meta-eis.h:30-33`,
`src/backends/meta-eis-client.c:226-228, 266, 363-377, 1156-1180` in 51.1).
The keyboard device gets Mutter's XKB keymap
(`meta-eis-client.c:256-300`). Consequently an EI client can send evdev
keycodes, but not UTF-8 text: characters outside the active XKB layout
cannot be produced. libei's own TEXT capability (keysym/UTF-8 events) was
added in commit `1f9be0f` (2025-08-14) and first shipped in **libei 1.6.0**
(`git tag --contains 1f9be0f` → `1.6.0`; Arch has 1.6.0-2). Mutter 51.1
does not handle `EIS_EVENT_TEXT*`, so that capability is not negotiated.

**Observed — portal handshake.** The supported app-facing path is
`org.freedesktop.portal.RemoteDesktop`, **version 2**:

1. `CreateSession(a{sv}) -> o` — returns a `Request`; the response contains
   `session_handle`.
2. `SelectDevices(o session, a{sv})` with `types = 1` (KEYBOARD), and
   optionally `persist_mode` and `restore_token`.
3. `Start(o session, s parent_window, a{sv})` — presents the user dialog
   unless a valid restore token is supplied; the response contains the
   granted `devices` mask and a new `restore_token`.
4. `ConnectToEIS(o session, a{sv}) -> h` — returns a Unix socket fd for
   `ei_setup_backend_fd()`; only after `Start`, once per session. After
   that, the `Notify*` D-Bus methods must not be used.

Spec: `data/org.freedesktop.portal.RemoteDesktop.xml` (installed
`/usr/share/dbus-1/interfaces/org.freedesktop.portal.RemoteDesktop.xml`,
xdg-desktop-portal 1.22.1): interface doc "version 2" (line 28), methods at
lines 145, 203, 262, 525; `ConnectToEIS` docs at 498-523; restore-token
rules at 106-120. GNOME backend implementation:
`https://gitlab.gnome.org/GNOME/xdg-desktop-portal-gnome/-/blob/main/src/remotedesktop.c`
(commit `fd1e79f`, 2026-11-12), `handle_connect_to_eis` at 1066-1107,
dialog creation at 786, restore path at 711-737.

**Observed — dialog.** GNOME's backend creates a GTK dialog titled "Remote
Desktop" with "Allow Remote Interaction", "Allow Clipboard Access",
"Remember This Selection" and a "Share" button
(`src/remotedesktopdialog.ui:7, 66-93`; `src/remotedesktop.c:786`). It is
shown on `Start` unless `restore_from_data()` succeeds.

**Observed — token semantics.** The token is **single-use**; each `Start`
returns a new one, and the app must store the new token. The spec: "The
restore token is invalidated after using it once. To restore the same
session again, use the new restore token sent in response to starting this
session." (`RemoteDesktop.xml:106-120`). `persist_mode=2` means "until
explicitly revoked". In practice the user consents once and later runs are
silent if the newest token is replayed: python-libei's dated test on a real GNOME Wayland
session (RemoteDesktop v2) recorded 5.4 s with dialog on first run and
0.2 s with no dialog on token replay
(`python-libei` docs/developers/verification.md, 2026-09-01).

**Observed — private alternative.** Mutter also exports
`org.gnome.Mutter.RemoteDesktop` (private API, version 1) with
`CreateSession`, `Start` and `ConnectToEIS`
(`data/dbus-interfaces/org.gnome.Mutter.RemoteDesktop.xml`; the XML says
"This API is private and not intended to be used outside of the integrated
system that uses libmutter"). Session methods only check that the caller is
the same bus peer that created the session
(`src/backends/meta-remote-desktop-session.c:411-417, 442-462`); the EIS
client code says "We're relying on some third party to filter clients for
us" (`meta-eis-client.c:1156`). **Inference:** a session-bus client could in
principle create a session and obtain an EIS fd without the portal dialog.
I did not test this on a live GNOME session; the interface is explicitly
unsupported.

### 2. Existing CLI tools and bindings

| Tool | Mechanism | Arch packaging | Types arbitrary Unicode? |
|---|---|---|---|
| `dotool` 1.6 | `/dev/uinput` | AUR `dotool 1.6-1` (17 votes) | No — only characters present in the active XKB layout (source: `dotool.go:358-371`; emits "impossible character for layout") |
| `ydotool` 1.0.4 | `/dev/uinput` via `ydotoold` | `extra 1.0.4-3` | No — ASCII-only 128-entry table (`Client/tool_type.c:42-134`) |
| `wtype` 0.4 | wlroots `zwp_virtual_keyboard_manager_v1` | `extra 0.4-2` | Yes, by uploading a synthetic keymap — but **not on GNOME**; Mutter has no such protocol (`main.c:568`) |
| `wshowkeys` | displays keys; wlroots layer-shell | AUR `wshowkeys-git` only | Not an injection tool |
| libei demo tools | `ei-demo-client`, `eis-demo-server`, `eis-demo-server-uinput`, `oeffis-demo-tool` | Arch `libei` installs only `/usr/bin/ei-debug-events` | `oeffis-demo-tool` does portal + EIS, but is a demo, not a typing CLI |
| `eitype` 0.2.2 | EI + XDG RemoteDesktop portal, token persistence | AUR `eitype 0.2.2-1`; PyPI wheels | No arbitrary Unicode — maps chars through the server keymap and errors with "Character not found in keymap" (`src/lib.rs:1358-1370`; `EiTypeError::CharNotFound`) |
| `python-libei` 0.8.2 | ctypes over libei/libeis/liboeffis + portal | PyPI only (2026-11-14) | `keyboard_key()` takes evdev keycodes; `text_utf8()` requires a TEXT-capable EIS (libei 1.6+), which Mutter is not |
| `pyxeis` | — | Not found on PyPI (404) or AUR | Not established that it exists |

`eitype` is the closest existing analogue to `wtype`/`ydotool` for GNOME: its
README states it uses the RemoteDesktop portal by default and works on GNOME
and KDE, and it saves the token at `~/.cache/eitype/restore_token`
(`README.md`, "Connection Methods"). Its GNOME support was not tested here.
`python-libei` documents the same limitation: keyboard input is key
positions, not characters (`README.md:52-56`; `docs/recipes.md:145-171`).

### 3. GNOME Shell's on-screen keyboard

**Observed.** The shell's `KeyboardController` creates a Clutter virtual
input device from the seat:

```js
const seat = global.stage.context.get_backend().get_default_seat();
this._virtualDevice = seat.create_virtual_device(Clutter.InputDeviceType.KEYBOARD_DEVICE);
```

`js/ui/keyboard.js:1856` (gnome-shell commit `7fcbec8`, 2026-11-09). It sends
key events with `notify_keyval()` (`keyboard.js:1994-1999`). For text it calls
`Main.inputMethod.commit(str)` (`keyboard.js:1980`), which is the shell's
`Clutter.InputMethod` subclass (`js/misc/inputMethod.js:24`), backed by IBus
(`inputMethod.js:73-86`). The shell registers this input method with Mutter
at startup (`js/ui/main.js:241`). Mutter's
`clutter_input_method_commit()` turns that into a text-input commit
(`clutter/clutter/clutter-input-method.c:282`;
`src/wayland/meta-wayland-text-input.c:337`).

- **From a GNOME Shell extension:** technically yes. Extensions are ES
  modules dynamically imported into the shell's own process
  (`js/ui/extensionSystem.js:505`), so they can call the same Clutter API.
  This is an inference from same-process execution; I did not test an
  extension doing it, and it is not a documented extension API.
- **From a normal application:** no. There is no Wayland protocol or D-Bus
  method that exposes `ClutterVirtualInputDevice` to clients. The supported
  app-facing path that reaches the same mechanism is the portal/EIS path,
  where Mutter's own `MetaRemoteDesktopSession` drives a
  `ClutterVirtualInputDevice` (`src/backends/meta-remote-desktop-session.c:100,
  189, 529-580`).

### 4. `dotool` as an alternative to `ydotool`

**Observed.** `dotool` writes to `/dev/uinput`; its manpage says it needs
write permission to `/dev/uinput`, granted to group `input` by its udev rule
(`80-dotool.rules`: `MODE="0620"`). No root is needed once the user is in
`input`. `ydotool` uses the same kernel interface through `ydotoold`;
Arch's `80-uinput.rules` grants group `input` mode `0660`, and the user
service runs `ydotoold` (`pacman -Ql ydotool`).

**Observed differences.**

- `dotool` is layout-aware: it uses libxkbcommon and the current XKB layout
  (`DOTOOL_XKB_LAYOUT`, `DOTOOL_XKB_VARIANT`) and handles dead keys
  heuristically (`doc/dotool.1.scd:32-46`, `dotool.go:358-371`). `ydotool`
  1.0.4's `type` uses a compiled-in US-ASCII table and has no layout
  handling (`Client/tool_type.c:42-134`; README "Custom keyboard layouts").
- `dotool` needs no daemon (optional `dotoold`/`dotoolc` for a persistent
  instance); `ydotool` has required `ydotoold` since 1.0.0.
- Neither types arbitrary Unicode (CJK/emoji) with `type`. `dotool` errors
  per character; `ydotool` silently drops/misreads non-ASCII.
- `dotool` is GPLv3; `ydotool` is AGPLv3.

**Inference:** both work on GNOME because they create a kernel input device
that Mutter/libinput sees as a normal keyboard; I did not test either on a
live GNOME 51.1 session.

### 5. Input-method paths

- **IBus engine:** available in principle to an ordinary app. An engine is a
  D-Bus service; `ibus_engine_commit_text()` emits a `CommitText` signal to
  ibus-daemon ("Commit output of input method to IBus client",
  `src/ibusengine.c:1982-1993`; API docs IBusEngine.html). It must be
  registered (runtime `RegisterComponent`, `src/ibusbus.h:658-666`; the
  user component directory is disabled in `src/ibusregistry.c:271-280`, so
  the normal install path is the system component directory) and the user
  must select it as the active input source. GNOME Shell starts ibus-daemon
  and is itself an IBus client (`js/misc/ibusManager.js:112`,
  `js/misc/inputMethod.js:73-86`), then forwards commits to the focused
  text-input client. **Inference:** commits can carry arbitrary UTF-8, but
  only while this engine is the active input source; it is not a silent
  background injection path. Activation of an engine without user selection
  was not established.
- **text-input-v3:** Mutter implements the compositor side
  (`src/wayland/meta-wayland-text-input.c`,
  `meta-wayland-versions.h:51`), and sends `commit_string` to the focused
  client (line 337). This is a **receive** path for apps; a client cannot use
  it to send text to another client.
- **zwp_input_method_v1/v2:** not implemented by Mutter. A search of the
  Mutter 51.1 tree finds no `zwp_input_method` symbol (0 files); GNOME uses
  IBus instead. Even on compositors that implement it, the compositor grants
  the input-method role; an ordinary app cannot take it unilaterally.

**Terminals — do they accept IME commits?** All four checked do, as clients:

- **VTE / gnome-terminal:** VTE connects the GtkIMContext "commit" signal and
  appends the text (`src/widget.cc:112-120, 1999-2011`, VTE 0.84.1).
  Arch's GTK3 ships `im-wayland.so`, which binds
  `zwp_text_input_manager_v3` (`modules/input/imwayland.c:40-41, 205` in
  gtk-3.24.52) and `im-ibus.so`; either module ends at the same commit
  signal.
- **kitty:** its bundled GLFW binds `zwp_text_input_manager_v3` and handles
  `commit_string` (`glfw/wl_text_input.c:15-16, 91, 137`); IME can be
  disabled with `wayland_enable_ime no` (kitty options definition).
- **alacritty:** 0.17.0 handles `WindowEvent::Ime(Ime::Commit(text))`
  (`src/event.rs:2019`) and enables IME (`src/display/window.rs:193`);
  winit 0.30.9's Wayland backend binds `zwp_text_input_manager_v3`
  (`src/platform_impl/linux/wayland/seat/text_input/mod.rs:9-11`). Known
  IME bugs remain (e.g. alacritty issues #7456, #8709).
- **foot:** binds `zwp_text_input_manager_v3` and adds a text-input listener
  (`wayland.c:146-160, 1638-1645`, foot 1.28.0).

## Observed vs inferred

**Observed:** Mutter 45.0+ contains an EIS server; Mutter 51.1's EIS seat has
no TEXT capability; the portal spec and GNOME backend implement
CreateSession/SelectDevices/Start/ConnectToEIS v2; the token is single-use;
the GNOME dialog exists; GNOME Shell uses ClutterVirtualInputDevice and an
IBus-backed Clutter.InputMethod; Mutter implements text-input-v3 but not
input-method-v1/v2; the package versions and tool source behaviour above.

**Inferred:** dotool/ydotool work on GNOME via uinput; a Shell extension can
call the Clutter API; the private Mutter D-Bus API may be callable by any
session-bus peer; an IBus engine's commits reach the focused client through
GNOME Shell; eitype works against Mutter 51.1.

## Open / not established

- No live GNOME session on this machine, so the portal dialog and EIS fd were
  not exercised here. The portal interface was not even exported in the
  running swayfx session.
- Whether Mutter's private `org.gnome.Mutter.RemoteDesktop` is reachable by
  an arbitrary app on a stock GNOME session (source suggests yes; untested;
  unsupported).
- Whether `eitype`/`wdotool` work against Mutter 51.1 in practice.
- Whether an IBus engine can be activated or can commit without the user
  selecting it as the input source.
- `pyxeis` could not be found; its existence is not established.

## Project note

`docs/platforms.md:394` says ydotool works "including eventually GNOME via
libei". That is wrong: ydotool 1.0.4 uses uinput and has no libei code. It
works on GNOME (via uinput) independently of libei. The README does correctly
say wtype is unsupported on GNOME (`docs/platforms.md:393`).
