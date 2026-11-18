# Can each injection backend type CJK directly?

Research note. Read-only; no code changed.

## Question

For each backend speakinput can use, can it *type* arbitrary Unicode (CJK)
characters directly, without clipboard + paste? Specifically: does it create
or remap a keycode whose keysym is the character's Unicode keysym, and does
the receiving terminal turn that keysym back into UTF-8?

## Local grounding

- Project at `/mnt/projects/speakinput`; backend selection and the ASCII /
  Unicode split are in `src/speakinput/injector.py:5-27` and
  `src/speakinput/injector.py:240-248`.
- Non-ASCII text currently goes to `_inject_unicode`
  (`src/speakinput/injector.py:250-275`), which uses the paste chord.
- Installed versions on this machine (Arch): `wtype 0.4-2`,
  `ydotool 1.0.4-3`, `pynput 1.8.2` in
  `.venv_archlinux/lib/python3.14/site-packages/pynput`
  (`pynput/_info.py:19`). PyPI reports 1.8.2 as the current release
  (uploaded 2026-05-12).

## Findings

### 1. `wtype` 0.4 — yes, arbitrary Unicode, by uploading a synthetic keymap

**Observed.** `wtype` does not remap one spare key in the user's layout. It
builds a *new* XKB keymap containing one keycode per distinct character and
uploads it to the compositor as the virtual keyboard's keymap
(`upload_keymap`, `main.c:452-500`). Text characters are converted with:

```c
// main.c:161-188
xkb_keysym_t xkb = xkb_utf32_to_keysym(ch);
...
return append_keymap_entry(wtype, ch, xkb);
```

and the keymap is emitted as

```c
// main.c:480-485
fprintf(f, "xkb_symbols \"(unnamed)\" {\n");
for (size_t i = 0; i < wtype->keymap_len; i++) {
    fprintf(f, "key <K%ld> {[", i + 1);
    print_keysym_name(wtype->keymap[i].xkb, f);
    fprintf(f, "]};\n");
}
```

with keycodes `9 .. keymap_len+8` (`main.c:466-473`). For `中` the keysym
name printed is `U4E2D` (libxkbcommon's `xkb_keysym_get_name` returns `U%04X`
for direct-encoded keysyms, `xkbcommon/src/keysym.c:66-92`). The keymap is
uploaded once, before
any key is sent (`main.c:534-535`), and each character is then sent as a
press/release of its keycode (`type_keycode`, `main.c:345-357`). wlroots adds 8
to the event keycode before the XKB lookup (`wlr_keyboard_notify_key`,
`wlroots/types/wlr_keyboard.c:93-106`), which is why keymap keycode `i+9`
matches the event key `i+1` that wtype sends.

So "not in the current XKB layout" is irrelevant: the character is in the
keymap wtype itself uploads. libxkbcommon maps every valid Unicode scalar
value except U+0000 and surrogates to a keysym
(`xkbcommon/src/keysym-utf.c:912-944`: Latin-1 1:1, C0 controls to legacy
keysyms, otherwise `ucs | 0x01000000`).

Confirmed use case in the project's own README: `wtype ∇⋅∇ψ = ρ`
(`https://github.com/atx/wtype/blob/v0.4/README.md`).

**Limitations / things that fail**

- Locale: text is decoded with `mbstowcs` after `setlocale(LC_CTYPE, "")`
  (`main.c:288-295`). In a non-UTF-8 locale wtype exits with
  "Failed to deencode input argv".
- U+0000 and surrogate code points get `XKB_KEY_NoSymbol` and produce no
  character (`xkb_utf32_to_keysym`; `main.c:177-178`).
- No explicit character cap in wtype. The cap is one keycode per *distinct*
  character (repeats reuse the entry, `main.c:171-175`); keycodes start at 9
  and grow. I compiled a keymap with `maximum = 300` using the installed
  xkbcommon 1.13.2 (`xkbcli compile-keymap`) and it was accepted. Behaviour
  above that, or in a particular compositor/client, was not established.
- wtype 0.4 is from 2022-01-27. Version 0.5 (2026-10-20) exists specifically
  to "Avoid control-key collisions when typing literal text" (PR #77): in
  Chromium, a character placed on evdev keycode 14 can act like Backspace.
  So on 0.4, literal text on low keycodes can be misinterpreted by some
  Chromium/Electron clients. This is not specific to CJK.
- Only compositors implementing `zwp_virtual_keyboard_manager_v1`; wtype
  exits with "Compositor does not support the virtual keyboard protocol"
  otherwise (`main.c:523-525`).

### 2. `ydotool` 1.0.4 — no. ASCII only, with an out-of-bounds read for non-ASCII

**Observed.** `ydotool type` has a hard-coded 128-entry table indexed by the
byte value:

```c
// Client/tool_type.c:42-74
static const int32_t ascii2keycode_map[128] = { ... };

// Client/tool_type.c:111-134
static void type_char(char c, bool delay) {
    int kdef = ascii2keycode_map[c];
    if (kdef == -1) return;
    ...
}
```

Input is read byte by byte (`tool_type.c:296-304`, and `:269-284` for
`--file`). UTF-8 bytes for CJK are >= 0x80; `char` is signed on x86-64 Linux,
so the index is negative and the read is out of bounds (undefined behaviour).
There is no branch that decodes UTF-8 and no Unicode keysym path.

1.0.4 has no keymap file at all: the v1.0.4 tree contains only
`Client/tool_*.c`, `Daemon/ydotoold.c`, docs; the installed Arch package
installs no `/usr/share/ydotool/keymap` (`pacman -Ql ydotool`). The mapping is
the compiled-in table above. (Whether some older 0.x release had a keymap file
was not investigated.)

**Corroboration.** Upstream issue #249, "Unicode characters not printing"
(open since 2024-07-27): `ydotool type "ඞ"` prints nothing, and a commenter
confirms "1.0.4 doesn't type non-latin characters either"
(`https://github.com/ReimuNotMoe/ydotool/issues/249`). Issue #275 reports the
out-of-bounds read, e.g. typing "Å" pressing Numpad 0/Insert
(`https://github.com/ReimuNotMoe/ydotool/issues/275`).

**Conclusion:** `ydotool type` cannot type CJK. The only Unicode-capable use
of ydotool is `ydotool key <evdev>:1 ...` against a layout or compose key that
already produces the character, which is not general.

### 3. pynput 1.8.2

#### 3a. macOS — yes, via `CGEventKeyboardSetUnicodeString`

**Observed.** `Controller.type` walks the string one character at a time
(`keyboard/_base.py:479-499`). A character that is not in the current layout's
keycode→string map gets `vk = None`:

```python
# keyboard/_darwin.py:120, 147-148
vk = self.vk or mapping.get(self.char)
...
if vk is None and self.char is not None:
    CGEventKeyboardSetUnicodeString(result, len(self.char), self.char)
```

The map is built with `UCKeyTranslate` over virtual keycodes 0-127
(`_util/darwin.py:190-199`), so `中` is not found and the Unicode-string path
is taken. The event is created with keycode 0 and then has its string
overridden (`keyboard/_darwin.py:134-148`).

**Length limit.** The per-call limit that projects hit is 20 characters,
truncating silently. It is not in the current `CGEvent.h` (the header documents
only the getter/setter, `MacOSX10.15.sdk/.../CGEvent.h:196-206`), but is
reported by enigo, which chunks its text at 20 and cites enigo-rs/enigo#68
(`enigo/src/macos/macos_impl.rs:315-318`). pynput sends one Python character
per event, so this limit does not bind for BMP CJK. Inference: for a non-BMP
character pynput passes `len(self.char) == 1` while the UTF-16 form is 2 units,
which would truncate the surrogate pair; I did not test this.

**Caveat from Apple's header:** "application frameworks may ignore the Unicode
string in a keyboard event and do their own translation based on the virtual
keycode" (`CGEvent.h:198-201`). Which macOS terminal emulators honour it was
not established by source; this is application-dependent by API contract.

#### 3b. X11 — a remap path exists, but delivery uses `XSendEvent`, not XTest

**Observed.** A `KeyCode` built from a character has `vk = None`
(`_base.py:47-50`, `from_char` at `_base.py:140-146`; `_resolve` at
`_base.py:570-595`). The X11 backend then:

1. Converts the character to a keysym: `ord(char) | 0x01000000` for anything
   above U+00FF (`_util/xorg.py:354-365`).
2. Fails normal resolution because the keysym is not in the current keyboard
   mapping (`_xorg.py:351-366`).
3. Borrows a keycode: finds an all-NoSymbol keycode in
   `get_keyboard_mapping(8, 255-8)`, writes the keysym into one of its first
   four levels with `change_keyboard_mapping`, and records it in `_borrows`
   (`_xorg.py:386-464`).
4. Sends the event with `_send_key`, which builds an `Xlib` KeyPress/KeyRelease
   and calls `window.send_event(...)` on the focused window (`_xorg.py:289-320`)
   — i.e. XSendEvent, **not** XTest. XTest is only used when `key.vk is not
   None` (`_xorg.py:244-253`).

It raises `InvalidCharacterException` only if no keycode or level can be
borrowed (`_resolve_borrowing` returns `None` on `TypeError`, `_xorg.py:463-464`;
`Controller.type` converts `InvalidKeyException`/`ValueError` into
`InvalidCharacterException`, `_base.py:492-499`). It does not silently drop
the character.

**Not established:** whether a given application accepts key events delivered
with `send_event=True`. That is the reason this path is weaker than XTest.
No source-level check of GTK/Qt/terminal acceptance was done.

#### 3c. Windows — yes for BMP, via `KEYEVENTF_UNICODE`

**Observed.** `_parameters` uses `VkKeyScanW` (`_util/win32.py:120-121`). If the
character is not reachable on the current layout, it falls back to a Unicode
`SendInput` record:

```python
# keyboard/_win32.py:81-96
elif ord(self.char) > 0xFFFF:
    raise ValueError
else:
    res = VkKeyScan(self.char)
    if (res >> 8) & 0xFF == 0:
        ...
    else:
        vk = 0
        scan = ord(self.char)
        flags = KEYBDINPUT.UNICODE
```

So `中` (U+4E2D, <= 0xFFFF) is sent as a `KEYEVENTF_UNICODE` scan code, which
the Win32 input model delivers to the focused window as `WM_CHAR` (inference
from the API, not tested here). Non-BMP characters raise `ValueError`, which
`type()` turns into `InvalidCharacterException` (`_base.py:492-499`), so emoji
do not work on Windows.

### 4. Wayland terminals and Unicode keysyms

**Observed, per terminal.** All five convert a keysym/Unicode keysym to UTF-8
and write it to the PTY:

- **foot**: `keyboard_key` gets `xkb_state_key_get_utf8`
  (`foot/input.c:1784-1806`) and `legacy_kbd_protocol` writes the bytes with
  `term_to_slave(term, utf8, count)` (`foot/input.c:1038`, `:1193`). foot
  recompiles the keymap on every `wl_keyboard.keymap` event
  (`foot/input.c:560-620`, compile at `:604`).
- **kitty**: its GLFW fork calls `xkb_state_key_get_utf8` into the event's
  text field (`kitty/glfw/xkb_glfw.c:1073-1075`, assigned at `:1122`). With
  the default (legacy) keyboard flags, `encode_glfw_key_event` returns
  `SEND_TEXT_TO_CHILD` for any event with non-control text
  (`kitty/key_encoding.c:458-467`), and `send_key_to_child` writes that text
  verbatim (`kitty/keys.c:212-215`). It recompiles the keymap from the
  compositor fd in `keyboardHandleKeymap` (`kitty/glfw/wl_init.c:540-560`).
- **alacritty**: uses winit's `KeyEvent::text_with_all_modifiers`
  (`alacritty/src/input/keyboard.rs:39`, `:262`), whose Wayland implementation
  is `xkb_state_key_get_utf8` (`winit 0.30.9`,
  `src/platform_impl/linux/common/xkb/mod.rs:324-329` and
  `state.rs:85-91`), and writes the bytes to the PTY (`keyboard.rs:86-101`).
- **VTE / GNOME Terminal**: `gdk_keyval_to_unicode(keyval)` then
  `g_unichar_to_utf8` (`vte/src/vte.cc:5619-5635`), sent at `:5672-5674`.
  GDK's Wayland seat sets the keyval from
  `xkb_state_key_get_one_sym` (`gtk/gdk/wayland/gdkseat-wayland.c:1176`), and
  `gdk_keyval_to_unicode` maps every `0x01000000 | codepoint` directly
  (`gtk/gdk/gdkkeyuni.c:889-903`). VTE's keymap is compiled from the
  compositor keymap (`gtk/gdk/wayland/gdkkeymap-wayland.c:610`).
- **Konsole**: `Vt102Emulation::sendKeyEvent` looks up the Qt key in the
  `.keytab`; with no entry and empty `entry.text()`, it sends
  `_encoder.encode(event->text())` (`konsole/src/Vt102Emulation.cpp:3282-3347`).
  Qt's Wayland keyboard sets that text with
  `QXkbCommon::lookupString`, i.e. `xkb_state_key_get_utf8`
  (`qtwayland/src/client/qwaylandinputdevice.cpp:1310-1317`;
  `qtbase/src/gui/platform/unix/qxkbcommon.cpp:407-416`); Konsole then encodes
  that QString with the profile's codec (`_encoder`).

**So:** at the source level, all five insert the character for a key event
whose keysym is U+4E2D. None of them filters to printable ASCII or requires
the keysym to exist in a physical layout; the keymap is whatever the
compositor sent.

**Important scope limit.** wtype's protocol is only available on compositors
that implement it. KWin does not: the KDE tracker has an open request
"Implement zwp_virtual_keyboard_manager_v1 protocol"
(`https://bugs.kde.org/show_bug.cgi?id=512996`). I found no file implementing
that protocol in a listing of either the Mutter or the KWin source tree, but
that is a tree/name check, not a full content search. So on GNOME or KDE,
wtype cannot run at all, and VTE/Konsole's behaviour under a Unicode keysym is
only reachable on a compositor that does implement the protocol (e.g. a
wlroots session running GNOME Terminal or Konsole).

## Observed vs inferred vs assumed

**Observed (read in source / run locally)**

- wtype 0.4 uploads a synthetic keymap, one keycode per distinct character,
  keycodes from 9; `xkb_utf32_to_keysym` is used; no explicit cap.
- ydotool 1.0.4 uses a 128-entry ASCII table, indexes it with a signed `char`,
  and has no keymap file; upstream issue #249 confirms non-Latin is not typed.
- pynput 1.8.2: macOS uses `CGEventKeyboardSetUnicodeString` per character;
  X11 borrows a keycode and uses XSendEvent; Windows uses
  `KEYEVENTF_UNICODE` for BMP and raises for non-BMP.
- foot, kitty, alacritty, VTE and Konsole all resolve a Unicode keysym to
  UTF-8 through `xkb_state_key_get_utf8` / `gdk_keyval_to_unicode` (or Qt's
  wrapper around the former) and write the bytes to the PTY.
- wlroots compiles the virtual keyboard's keymap and re-broadcasts it to
  clients before keys are processed (`wlroots/types/wlr_virtual_keyboard_v1.c:53-75`,
  `wlroots/types/seat/wlr_seat_keyboard.c:349-365`).
- xkbcommon 1.13.2 locally compiles a keymap with `maximum = 300`.

**Inferred**

- pynput's macOS astral-character path truncates a surrogate pair because
  `len(self.char)` is a code-point count, not a UTF-16 unit count.
- wtype fails cleanly in a non-UTF-8 locale because `mbstowcs` returns -1.
- wtype's Unicode keysyms work in a terminal only if the terminal rebuilds its
  xkb state from the compositor's keymap; verified for foot, kitty, Qt and GDK,
  inferred for VTE/Konsole from those toolkits.
- ydotool's negative index produces garbage key events (consistent with the
  "prints gibberish" reports) rather than a clean no-op.

**Assumed**

- The user's locale is UTF-8 on Wayland.
- "Current pynput" means 1.8.2, the installed and current PyPI release.
- The compositor in use for wtype is wlroots-based (or another implementation
  of `zwp_virtual_keyboard_manager_v1`).

## Open / conflicting

- **pynput X11 delivery.** The mechanism exists, but XSendEvent key events are
  not equivalent to XTest events, and the client must also have processed the
  `ChangeKeyboardMapping` before the synthetic event. Whether a specific
  terminal accepts this was not established. This needs an on-target test.
- **macOS app behaviour.** Apple's own header warns that frameworks may ignore
  the Unicode string; which terminal emulators do was not established.
- **Very large keycodes in wtype.** xkbcommon accepts high keycodes, but no
  compositor or client test was done for texts with many distinct characters.
- **wtype 0.4 vs 0.5.** The installed version is 0.4; 0.5 contains a fix for
  keycode collisions in Chromium-based clients. Any conclusion about 0.4 in
  such clients may not hold for 0.5.
- **Emoji / non-BMP.** Not established for wtype or for the terminals. This is
  outside the CJK (BMP) case.
- **Mutter.** The evidence that Mutter lacks the protocol is a tree/name check
  plus secondary reports; I did not read its protocol dispatch table.

## Recommendation for the decision

The evidence supports a direct-typing path for CJK on three of the four
backends:

- `wtype`: yes, for any valid BMP (and by construction non-BMP) Unicode
  scalar, on compositors that implement the virtual-keyboard protocol.
- pynput on macOS and Windows: yes for BMP CJK, via the platform's Unicode
  injection API; the macOS path depends on the target app honouring
  `CGEventKeyboardSetUnicodeString`, and Windows rejects non-BMP.
- pynput on X11: the remap-and-send mechanism exists, but delivery is by
  XSendEvent, so it is the weakest of the three and needs a real-target check.
- `ydotool` 1.0.4: no. It cannot type CJK at all; do not route CJK through it.
- The terminal side is not the blocker: all five terminals decode Unicode
  keysyms to UTF-8. The blocker for GNOME Terminal/Konsole is the compositor
  (Mutter/KWin do not implement the protocol wtype needs), not the terminal.
