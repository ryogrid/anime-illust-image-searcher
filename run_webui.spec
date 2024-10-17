# -*- mode: python ; coding: utf-8 -*-
import site
import os

block_cipher = None

assert len(site.getsitepackages()) > 0

# Choose path containing "site-packages"
package_path = site.getsitepackages()[0]
for p in site.getsitepackages():
    if "site-package" in p:
        package_path = p
        break

a = Analysis(
    ['run_webui.py'],
    pathex=[],
    binaries=[],
    # copy streamlit files
    datas=[(os.path.join(package_path, "altair/vegalite/v5/schema/vega-lite-schema.json"),
        "./altair/vegalite/v5/schema/"),
       (os.path.join(package_path, "streamlit/static"),"./streamlit/static"),
       (os.path.join(package_path, "streamlit/runtime"),"./streamlit/runtime"),
       ],
    hiddenimports=[
       'gensim',
       'gensim.models',
       'gensim.models.lsimodel',
       'gensim.similarities',
       'numpy',
    ],
    hookspath=['./hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='run_webui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='run_webui',
)
