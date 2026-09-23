# ケース2: 履歴を遡った方向の反転

> たたき台。納得いかんところは書き換えてください ― このファイルがこのケースの唯一の仕様です。
> 行の形・queue と program の規則・返事の書き方・禁止事項は `scripts/llm/HANDWRITTEN.md` §1〜§5。

## 症状

```
人   後ろ向きに10m進もう、そしたら、バク転！   → [move backward 10m, backflip]   ○
（完了）
人   さっきと、逆方向に5mすすもう              → [move left 5m]                  ×
                                               正解は [move forward 5m]
```

方向を指す語が発話に無く、**履歴の中の move を見つけて、その向きを裏返す**必要がある。
データセット全体で13例しかなく、うち履歴だけが頼りのものは7例。だから当てずっぽうになる。

## 正解の決め方

教えたいのは2つ。どちらも**履歴の中の move を見つけて裏返す**話で、違うのは拾う数。

### ① 1手だけ裏返す ― 「さっきと逆方向に」

1. 「逆／反対」が指している対象を、**人の発話だけを遡って**特定する。直前の move が普通だが、
   「2つ前の」「さっきの◯◯」と言われたらそれ。
2. その向きを裏返す。`forward⇄backward`、`left⇄right`。旋回なら `turn left⇄turn right`、
   横回転なら `sideflip_left⇄sideflip_right`、前転⇄後転は `frontflip⇄backflip`。
3. 量は、言われていればその量。言われていなければ**元と同じ量**（「逆に同じだけ」）。
4. queue が空なら返すのは新しい1手。queue に何か載っているなら
   `HANDWRITTEN.md` §2 の表どおり（今すぐなら先頭に、終わったらなら末尾に）。

**裏返すのは向きだけ。** 速度も量も、言われていない限り元のまま引き継ぐ。

### ② まとめて巻き戻す ― 「元の場所に戻って」

来た道をそのまま逆再生する。**履歴の移動と旋回を逆順に並べ、それぞれの向きを裏返す。**

```
履歴  「前に3m」         [move forward 3.0m]
      「左に90度」       [turn left 90]
      「前に2m、バク転」 [move forward 2.0m, backflip]

人    元の場所に戻って
→     [move backward 2.0m, turn right 90, move backward 3.0m]
       最後にやった移動から順に、向きを裏返して巻き戻す
```

1. 拾うのは **move と turn だけ**。技（flip）と立ち（stance）はその場の動作なので落とす。
2. **順番を逆にする。** 最後にやった移動が、戻る手順の最初に来る。
3. 各手順の向きを裏返す。量と速度は元のまま。
4. 「半分だけ戻って」のように割合が言われたら、各手順の量をその割合にする。
5. 履歴に移動が1つも無ければ戻る先も無い。聞き返して `program` は queue のコピー。

言い方: 元の場所に戻って / さっきのとこまで戻って / 来た道戻って / 元いたとこ /
スタート地点に / 巻き戻して / 最初の位置に / 戻ってきて。

## 会話の作り方

- 1ターン目は普通の指示（移動・技・旋回の組み合わせ）。2ターン目以降で反転を出す。
- **反転を言う位置をばらす。** 2ターン目だけでなく、3〜5ターン目で「2つ前のやつと逆に」も作る。
- 反転の言い方を毎回変える: 逆方向に / 逆向きで / 反対側に / さっきと逆で / あべこべに /
  逆のほうへ / 今度は反対 / さっきの逆な。表に無い言い方を自分で足してよい。
- 実行中（queue が空でない）の反転も作る ― 「今走ってるの、逆向きにして」。
- **①と②を半々に。** ②は履歴が2手以上ないと成立しないので、3〜5ターンの会話にする。
- ②では旋回を混ぜた履歴も作る ― 巻き戻すときに `turn` の向きも裏返るのが肝。

## 例

```json
{"id": "reverse-0000", "category": "case_reverse", "style": "kansai",
 "turns": [
   {"role": "user", "content": "後ろに10m下がってからバク転や", "queue": []},
   {"role": "assistant", "reply": "ええで、下がってからバク転いくわ。",
    "program": [{"skill": "move", "dir": "backward", "speed": "normal", "distance_m": 10.0},
                {"skill": "flip", "kind": "backflip"}]},
   {"role": "user", "content": "ほな今度は逆方向に5m", "queue": []},
   {"role": "assistant", "reply": "逆や。前に5m行くで。",
    "program": [{"skill": "move", "dir": "forward", "speed": "normal", "distance_m": 5.0}]}
 ]}
```

②「元の場所に戻って」の例:

```json
{"id": "reverse-0002", "category": "case_reverse", "style": "polite",
 "turns": [
   {"role": "user", "content": "前に3m進んでください", "queue": []},
   {"role": "assistant", "reply": "前に3m進みます。",
    "program": [{"skill": "move", "dir": "forward", "speed": "normal", "distance_m": 3.0}]},
   {"role": "user", "content": "そこで左に90度、それから前に2m", "queue": []},
   {"role": "assistant", "reply": "左を向いてから2m進みます。",
    "program": [{"skill": "turn", "dir": "left", "angle_deg": 90.0},
                {"skill": "move", "dir": "forward", "speed": "normal", "distance_m": 2.0}]},
   {"role": "user", "content": "やっぱり元の場所に戻ってもらえますか", "queue": []},
   {"role": "assistant", "reply": "来た道を巻き戻しますね。",
    "program": [{"skill": "move", "dir": "backward", "speed": "normal", "distance_m": 2.0},
                {"skill": "turn", "dir": "right", "angle_deg": 90.0},
                {"skill": "move", "dir": "backward", "speed": "normal", "distance_m": 3.0}]}
 ]}
```
