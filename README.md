# MonoCam Motion Trace（MMT）

> [!IMPORTANT]
> **本ソフトウェアで制作したモーションは、商用・ポルノ・軍事・監視の目的には利用できません。**
> 推定に使う NLF の学習済みモデルは「非商用の研究用途」、それに含まれる SMPL の人体モデルは「非商用の研究・教育・芸術作品」に限って公開されています。
> 
> Motions generated using this software **may not** be used for commercial, pornographic, military, or surveillance purposes.
> The Neural Localizer Fields (NLF) pre-trained model used for estimation is licensed strictly for non-commercial research purposes, and the SMPL body model included within it is restricted exclusively to non-commercial research, educational, and artistic works.

単眼カメラで撮影した動画を、NLFを用いて姿勢を推定し、ダンスモーションとして調整するソフトウェアです。作成したダンスモーションは、MMD（MikuMikuDance）用のモーション（VMD）として出力できます。

## NLF（Neural Localizer Fields）
論文: [Neural Localizer Fields for Continuous 3D Human Pose and Shape Estimation](https://arxiv.org/abs/2407.07532) 

著者: István Sárándi, Gerard Pons-Moll

[Project Page](https://istvansarandi.com/nlf/) | [Github Repository](https://github.com/isarandi/nlf)

## 動作環境

| 項目 | 内容 |
|---|---|
| OS | Windows 10 / 11（64 ビット） |
| GPU | NVIDIA の GPU が必要（GeForce GTX 900 番台以降・ビデオメモリ 4 GB 以上・CUDA 12.8 に対応したドライバー）|
| ディスク | 約 16 GBほど |
| ネットワーク | 初回のキャプチャのみ、実行環境と学習済みモデルをダウンロードします |

## インストール

[BowlRoll](https://bowlroll.net/file/362078) からインストーラーをダウンロードしてください。

## 使い方

1. **モデルを読み込む**（ファイル → モデルを読み込む。PMX / PMD）
   モーションはモデルの体格（脚の長さ・腕の太さ・剛体など）に合わせて作るので、動画より先に読み込んでください。
2. **動画を読み込む**（ファイル → 動画を読み込む。mp4 / mov / webm）
   動画を読み込んだら、設定はデフォルトのまま「適用」ボタンを押してください。細かく調整したい場合は、必要に応じて設定を変更してください。
3. キャプチャが終わると、タイムラインに楽曲波形・モーション・足ピタ・手のポーズが入ります。
   必要に応じてタイムライン上で編集を行い、画面右上の「VMD出力」でモーションを書き出します
4. **MMD（MikuMikuDance）でモーション（VMD）を読み込み、微調整してください。**
   全てが正確に推定されるわけではないため、MMDでの微調整は必須です。また、躍動感あるダンスにするためには、人の力によるアレンジが不可欠です。このソフトウェアは、これまでのモーション制作を置き換えることを目的としません。

> [!TIP]
> MMD（MikuMikuDance）に備わっている、不要ﾌﾚｰﾑ削除（編集 → 不要フレーム削除）を使うことで、不要なキーを自動で削除できます。

## うまくいかないとき

| 症状 | 対処 |
|---|---|
| キャプチャ・変換が毎回失敗する | ファイル →「実行環境を再セットアップ」（実行環境を入れ直し、学習済みモデルが壊れていないか確かめます） |
| キャプチャが遅い | 推定する fps・解像度を下げる。精度を「速度優先」にする |
| GPU のメモリが足りない | 推定の詳細設定でバッチサイズを下げる。解像度を 480p にする |
| 人物が検出されない・別の人に移る | 人物検出のしきい値を下げる。対象の人物を「画面の中央の人」にする |
| モデルが後ろを向く | 変換の設定の「体の向きを反転する」 |
| 座る・寝転ぶ動きで体が浮く・沈む | 変換の設定の「浮いた体を床に戻す」「前後の傾きを補正する」を外す |
| 手・指先が体に入り込む | 変換の設定の「服の厚み」を大きくする |
| 足が浮遊する | タイムラインにある「足ピタ」を活用する |
| 足首に違和感がある | MMD（MikuMikuDance）で「足首」や「足首D」の角度、「つま先IK」の位置を調整する |

## 紹介動画
- [ニコニコ動画](https://www.nicovideo.jp/watch/sm46951485)
- [Youtube](https://www.youtube.com/watch?v=kr07l8K0MPk)

## ライセンス

本ソフトウェアのソースコードのライセンスは、 [GNU Affero General Public License v3.0](LICENSE) です。

学習済みモデル（NLF・MediaPipe Hands・Demucs・Allosaurus）と PyTorch は本ソフトウェアに同梱せず、初回のキャプチャのときに各配布元からダウンロードします。本ソフトウェアを使用する場合には、各配布元のライセンスが適用されます。

本ソフトウェアで制作した作品を公開するときは、次のクレジットを記載してください。
```
Neural Localizer Fields / István Sárándi, Gerard Pons-Moll
```
次のクレジットは任意ですが、こちらも一緒に記載していただけると開発者の励みになります。
```
MonoCam Motion Trace / yamak493
```

## 免責事項

本ソフトウェアは、現代のAI技術（NLFやMediaPipeなど）でどれくらいの精度のダンスモーションを生成できるかを体験することを目的としています。本ソフトウェアで制作した作品を、利用条件の範囲内で投稿・公開できることを確認はしていますが（[isarandi/nlf#51](https://github.com/isarandi/nlf/issues/51)）、開発者は本ソフトウェアに起因して発生した損害やトラブル（第三者との間の争いを含みます）について、理由の如何を問わず一切の責任を負いません。制作した作品の投稿・公開は、利用者の責任で行ってください。

