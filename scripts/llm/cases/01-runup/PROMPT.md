# ケース1: 助走の長さを言わない「〜ながら技」

> たたき台。納得いかんところは書き換えてください ― このファイルがこのケースの唯一の仕様です。
> 行の形・queue と program の規則・返事の書き方・禁止事項は `scripts/llm/HANDWRITTEN.md` §1〜§5。

## 症状

```
人   右方向に進みながら、途中で、右側転を！
Go2  [move right normal]                      ×  技が消えた
     [move right normal, sideflip_right]      ×  長さの無い move の後ろに手順は置けない（コンパイル不能）
     正解は [move right 3.0s, sideflip_right, move right 3.0s]
```

「進みながら」を**長さ無しの move（＝止めるまでずっと）**と読んでしまう。助走の長さを言わない
言い方は学習データの4%しかなく、横方向では250本中7本しかない。

## 正解の決め方

**「〜しながら、途中で技」は、移動が主で、技はその途中に挟まる。** 移動 → 技 → 移動の**3手で固定**。

```
右に進みながら、途中で右側転  →  [move right 3.0s, sideflip_right, move right 3.0s]
                                  進む → 途中で回る → 続きを進む
```

1. 長さが言われていなければ、**前も後ろも 3.0 秒**。これは読み手が補う既定値で、往復パーサも
   同じ値を入れる（`DATASET.md` §4）。勝手に 1m や 10 秒にしない。
2. 量が言われていればその単位のまま ― 「右に5m進みながら途中で側転」なら `distance_m` で書く。
   時間に寄せない。
3. 速度が言われていなければ `normal`。**前後の move は同じ向き・同じ速度。**
4. 走り技は向きと種類が対応する ― forward→frontflip / backward→backflip /
   left→sideflip_left / right→sideflip_right。**方向と合わない技は書かない。**
5. 「そのあともずっと」と明に言われたときだけ、3手目を長さ無しにする
   `[move 3.0s, flip, move∞]`。長さ無しは**必ず最後**。

**長さの無い move の後ろには何も置けない。** 「ずっと進みながら技」と言われても、前の move に
長さを与えるのが正解であって、`[move∞, flip]` は不正なプログラム。

## 会話の作り方

- **4方向すべてを均等に。** forward に偏らせない。横（left/right）と backward を厚く。
- 言い方を毎回変える: 進みながら / 行きながら / 動きつつ / 歩きながら / 走りながら /
  進んでる途中で / 途中で / その勢いで / 止まらんと / 助走つけて。表に無い言い方を足してよい。
- 「途中で」は現行データに技と組んだ例が**ゼロ**。意識して入れる。
- 2ターン目以降で出す形も作る ― 実行中に「そのまま途中で前転も入れて」。

## 例

```json
{"id": "runup-0000", "category": "case_runup", "style": "kansai",
 "turns": [
   {"role": "user", "content": "右に進みながら、途中で右側転いったって!", "queue": []},
   {"role": "assistant", "reply": "右に流れて、途中で横回転挟むで。",
    "program": [{"skill": "move", "dir": "right", "speed": "normal", "duration_s": 3.0},
                {"skill": "flip", "kind": "sideflip_right"},
                {"skill": "move", "dir": "right", "speed": "normal", "duration_s": 3.0}]},
   {"role": "user", "content": "ほな、そのあとは止まらんとずっと右な", "queue": []},
   {"role": "assistant", "reply": "回ったあとは右に行きっぱなしにするわ。",
    "program": [{"skill": "move", "dir": "right", "speed": "normal", "duration_s": 3.0},
                {"skill": "flip", "kind": "sideflip_right"},
                {"skill": "move", "dir": "right", "speed": "normal"}]}
 ]}
```
