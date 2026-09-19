# go2_nav_ws 依存リポジトリ用パッチ

`~/isaacsim/go2_nav_ws/src` に依存リポジトリを clone した直後に適用する。
upstream を pull し直したときも再適用すること(適用済みなら `git apply` は
"patch does not apply" で止まるだけで壊れない)。

## unitree_ros2-msg-build-deps.patch

対象: https://github.com/unitreerobotics/unitree_ros2

`unitree_go` / `unitree_api` の package.xml に
`<build_depend>rosidl_generator_dds_idl</build_depend>` を追加する。
両パッケージの CMakeLists は `find_package(rosidl_generator_dds_idl REQUIRED)`
しているのに package.xml に依存宣言がなく、colcon がビルド順を保証できない
(rosidl_dds と並列に走って find_package が失敗する)。

```bash
cd ~/isaacsim/go2_nav_ws/src/unitree_ros2
git apply <unitree_rl_lab>/ros2/patches/unitree_ros2-msg-build-deps.patch
```

## invariant-ekf-cmake-colcon.patch

対象: https://github.com/inria-paris-robotics-lab/invariant-ekf

1. `cmake_minimum_required` を 3.10 → 3.22 に引き上げ。
   FetchContent で取得される最新の jrl-cmakemodules がメインプロジェクトに
   CMake >= 3.22 の宣言を要求するため。
2. package.xml の `<build_type>` を `ament_cmake` → `cmake` に修正。
   CMakeLists は素の CMake で `ament_package()` を呼ばないため、ament_cmake
   宣言のままだと colcon が生成する package.dsv が存在しない
   `share/inekf/local_setup.bash` を参照し、`source install/setup.bash` の
   たびに "not found" 警告が出る。

```bash
cd ~/isaacsim/go2_nav_ws/src/invariant-ekf
git apply <unitree_rl_lab>/ros2/patches/invariant-ekf-cmake-colcon.patch
```
