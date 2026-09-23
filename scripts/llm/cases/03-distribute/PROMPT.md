# ケース3: 数の配分（「AとBを2ずつ」）

> たたき台。納得いかんところは書き換えてください ― このファイルがこのケースの唯一の仕様です。
> 行の形・queue と program の規則・返事の書き方・禁止事項は `scripts/llm/HANDWRITTEN.md` §1〜§5。

## 症状

```
人   後ろ足倒立を5秒、その後に、バク転、ハンドスプリングを2ずつ
Go2  [hindstand 5s, backflip, frontflip, frontflip]        ×  2が最後の技にしか掛かっていない
     正解は [hindstand 5s, backflip, backflip, frontflip, frontflip]
```

「ずつ／それぞれ／各N」を含む指示文はデータセット全体で3件しかなく、**3件とも「一個ずつ」**。
配分する数が2以上の例を一度も見ていない。一方、同じ技が2連続するプログラムは737件あるので、
**1つの技に「2回」を掛けるのは覚えている**。覚えていないのは、数が直前に並べた技**すべて**に
掛かるという読み方。

## 正解の決め方

教えたいのは2つ。どちらも**数が、並べた技すべてに掛かる**話で、違うのは並べる順。

### ① まとめて ― 「AとBを2ずつ」

`[A]*N + [B]*N`。**A を N 回やってから B を N 回。**

```
バク転と前転を2ずつ  →  [backflip, backflip, frontflip, frontflip]
```

### ② 交互に ― 「AとBを交互に2回ずつ」

`[A, B]*N`。**A B を N 周。**

```
前転と後転を交互に2回ずつ  →  [frontflip, backflip, frontflip, backflip]
```

「交互に」「かわりばんこに」「互い違いに」「順番に繰り返して」「AB AB で」と言われたら②。
何も言われなければ①。

### どちらにも共通

1. 3つ以上並んでも同じ ― ①は `[A]*N + [B]*N + [C]*N`、②は `[A, B, C]*N`。
2. 「それぞれ／各N／両方とも／どっちもN回」も数の掛かり方は同じ。
3. **1ステップ1回転。** 2回は2ステップ。`{"skill": "flip", ...}` を並べる。
   回数フィールドは存在しない。
4. 技の呼び名の対応 ― ハンドスプリング／前転／前まわり／前方宙返り＝`frontflip`、
   バク転／後転／後ろまわり＝`backflip`、横回転／側転＝`sideflip_left` か `sideflip_right`。
5. 立ち技（倒立・逆立ち）に掛かる場合も同じく N ステップ並べる。

**配分が掛かる範囲に注意。** 「倒立5秒、そのあとバク転と前転を2ずつ」の「2ずつ」は
**バク転と前転にだけ**掛かる。倒立は5秒のまま1回。

## 会話の作り方

- **①と②を半々に。** 「交互」の言い方も毎回変える。
- N は 1〜3 を混ぜる。**2と3を厚く**（1は既存データにある）。
- 並べる技は2つだけでなく3つの行も作る。②で3つなら `[A, B, C]` を N 周。
- 立ち技・旋回・移動が混ざった列の中に配分が現れる形も作る。
- 言い方を変える: 2ずつ / 2回ずつ / それぞれ2回 / 各2回 / 両方とも2回 / どっちも2発 /
  2連発ずつ / 一個ずつ。表に無い言い方を足してよい。
- 2ターン目以降で出す形も作る ― 「さっきの2つ、もう一回ずつ」。

## 例

```json
{"id": "distribute-0000", "category": "case_distribute", "style": "polite",
 "turns": [
   {"role": "user", "content": "後ろ足立ちを5秒、その後にバク転とハンドスプリングを2ずつお願いします", "queue": []},
   {"role": "assistant", "reply": "後ろ足で5秒立ってから、バク転2回、ハンドスプリング2回いきます。",
    "program": [{"skill": "stance", "kind": "hindstand", "duration_s": 5.0},
                {"skill": "flip", "kind": "backflip"}, {"skill": "flip", "kind": "backflip"},
                {"skill": "flip", "kind": "frontflip"}, {"skill": "flip", "kind": "frontflip"}]},
   {"role": "user", "content": "最後に前に3m", "queue": []},
   {"role": "assistant", "reply": "前に3m進みます。",
    "program": [{"skill": "move", "dir": "forward", "speed": "normal", "distance_m": 3.0}]}
 ]}
```

②「交互に」の例:

```json
{"id": "distribute-0001", "category": "case_distribute", "style": "kansai",
 "turns": [
   {"role": "user", "content": "前転と後転、交互に2回ずつやってみ", "queue": []},
   {"role": "assistant", "reply": "前、後ろ、前、後ろの順でいくで。",
    "program": [{"skill": "flip", "kind": "frontflip"}, {"skill": "flip", "kind": "backflip"},
                {"skill": "flip", "kind": "frontflip"}, {"skill": "flip", "kind": "backflip"}]},
   {"role": "user", "content": "ほな最後、左に90度", "queue": []},
   {"role": "assistant", "reply": "左に90度回るわ。",
    "program": [{"skill": "turn", "dir": "left", "angle_deg": 90.0}]}
 ]}
```
