# -*- mode: python -*-

block_cipher = None


a = Analysis(['SkipThisScene.py'],
             pathex=['C:\\Users\\ndrs\\Downloads\\models-master\\research\\object_detection\\censor\\App'],
             binaries=[],
             datas=[('C:\\Users\\ndrs\\Downloads\\models-master\\research\\object_detection\\censor\\App\\venv_app\\Lib\\site-packages\\tensorflow\\python\\_pywrap_tensorflow_internal.pyd','tensorflow\\python')],
             hiddenimports=['pandas','numpy'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='SkipThisScene',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True , icon='img/icon.ico')
