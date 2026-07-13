# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for speakinput.

Builds a single-file frozen binary. PyInstaller's default analysis should
pick up pywhispercpp's bundled native libs (.dylibs on macOS, .libs on
Linux) automatically because they sit next to the Python module on
sys.path. The hidden imports below cover libraries that do dynamic
imports (rumps, AppKit, pynput's platform backends) which PyInstaller's
static analysis misses.

Run via `release.sh`, not directly.
"""

from pathlib import Path

block_cipher = None

# Modules imported dynamically that PyInstaller's static analysis misses.
# Listed once here so the build is reproducible — adding them via the CLI
# in release.sh would silently rot.
HIDDEN_IMPORTS = [
    # macOS menu-bar indicator (optional extra). Importing it requires
    # the rumps + AppKit + Foundation dance; without this hint the
    # bundle won't be runnable on a Mac that has rumps installed.
    "rumps",
    "AppKit",
    "Foundation",
    # pynput uses platform-specific backends picked at runtime. The
    # darwin backend imports HIToolbox, the X11/Wayland backends
    # import dbus. List both — extras the host doesn't need get
    # tree-shaken by the linker.
    "pynput.keyboard._darwin",
    "pynput.mouse._darwin",
    "pynput.keyboard._xorg",
    "pynput.mouse._xorg",
    # sounddevice pulls in _sounddevice via ctypes at runtime.
    "sounddevice",
    # pywhispercpp's example directory adds a top-level `pywhispercpp.examples`
    # module that some libraries import speculatively. Skip — it's not used
    # by the app, and including it would pull numpy + whisper CLI into the
    # frozen binary needlessly.
]


a = Analysis(
    ["src/speakinput/__main__.py"],
    pathex=["src"],
    binaries=[],
    datas=[],
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Trim the fat: these aren't used by speakinput and inflate
        # the binary by 30+ MB.
        "tkinter",
        "test",
        "unittest",
        "pydoc_data",
        "setuptools",
        "pip",
        "wheel",
        "email",
        "html",
        "http",
        "xml",
        "xmlrpc",
        "pydoc",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="speakinput",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX-compressed binaries sometimes trip macOS notarization
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,  # let the host decide
    codesign_identity=None,
    entitlements_file=None,
)
