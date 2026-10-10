"""python -m mmt_engine: アプリから起動されるエンジンのサーバー。"""
import os
import sys

# 数値計算のスレッド数は OS に任せる。tflite・OpenMP の重複したライブラリの読み込みで止まらないようにする
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--selftest':
        from .selftest import main as selftest_main
        sys.exit(selftest_main())
    if len(sys.argv) > 1 and sys.argv[1] == '--prefetch':
        from .selftest import prefetch
        sys.exit(prefetch())
    from .server import main
    main()
