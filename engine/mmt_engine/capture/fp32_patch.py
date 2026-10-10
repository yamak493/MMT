"""NLF の TorchScript を単精度（float32）で動かす版を作る。

NLF は推論を半精度（float16）で行う。GeForce GTX 16xx（Turing TU11x）などの GPU では半精度の畳み込み・行列積が
NaN になり、人物が 1 人も検出されない。TorchScript の中で半精度へ変換している所は数か所だけなので、それを
単精度への変換に書き換えたファイルを作り、読み込んだ後に重みも単精度にする（メモリは約 2 倍、速度は GPU しだい）。

TorchScript のファイルは zip で、code/ 以下に Python に近いソース（dtype は番号: 5 = float16、6 = float32）がある。
"""
import os
import zipfile
from pathlib import Path

# (ファイル名の末尾, 置き換える前, 置き換えた後)
PATCHES = (
    ('nlf/pt/multiperson/multiperson_model.py', 'torch.to(im, 5)', 'torch.to(im, 6)'),
    ('nlf/pt/multiperson/multiperson_model.py', 'torch.eq(ops.prim.dtype(im), 5)', 'torch.eq(ops.prim.dtype(im), 6)'),
    ('nlf/pt/multiperson/person_detector.py', 'forward(torch.to(images, 5), )', 'forward(torch.to(images, 6), )'),
    ('nlf/pt/models/nlf_model.py', 'transpose_weights(weights, backbone_link_dim, 5, )',
     'transpose_weights(weights, backbone_link_dim, 6, )'),
    ('nlf/pt/models/nlf_model.py', 'transpose_weights(weights_fl, backbone_link_dim0, 5, )',
     'transpose_weights(weights_fl, backbone_link_dim0, 6, )'),
)


def fp32_path(src):
    src = Path(src)
    return src.with_name(src.stem + '.fp32' + src.suffix)


def make_fp32_model(src):
    """書き換えた TorchScript（<名前>.fp32.torchscript）を作る（既にあれば作らない）。戻り値はそのパス。"""
    src = Path(src)
    dst = fp32_path(src)
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    tmp = dst.with_name(dst.name + '.part')
    applied = set()
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(tmp, 'w', zipfile.ZIP_STORED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if '/code/' in item.filename and item.filename.endswith('.py'):
                text = None
                for i, (suffix, before, after) in enumerate(PATCHES):
                    if item.filename.endswith(suffix):
                        text = (text if text is not None else data.decode('utf-8'))
                        if before in text:
                            text = text.replace(before, after)
                            applied.add(i)
                if text is not None:
                    data = text.encode('utf-8')
            info = zipfile.ZipInfo(item.filename, date_time=item.date_time)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = item.external_attr
            zout.writestr(info, data)
    if len(applied) != len(PATCHES):
        os.remove(tmp)
        raise RuntimeError('NLF のモデルを単精度に書き換えられませんでした（想定していない版のモデルです）')
    os.replace(tmp, dst)
    return dst
