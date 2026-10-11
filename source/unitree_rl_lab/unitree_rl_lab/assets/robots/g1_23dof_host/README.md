# G1 23dof model from HoST

`g1_23dof.urdf` is copied unchanged from [OpenRobotLab/HoST](https://github.com/OpenRobotLab/HoST)
(`legged_gym/resources/robots/g1`, commit 70bb580, MIT). `g1_23dof_nomarkers.urdf` is the same model
without HoST's massless keyframe / auxiliary marker links (see the comment at the top of that file).

The meshes are not in this repository (`*.STL` is git-ignored). Copy them from HoST before using the
`Unitree-G1-23dof-GetUp-HoSTModel-v0` / `-HoSTURDF-v0` tasks:

```bash
git clone --depth 1 https://github.com/OpenRobotLab/HoST.git /tmp/HoST
cp -r /tmp/HoST/legged_gym/resources/robots/g1/meshes source/unitree_rl_lab/unitree_rl_lab/assets/robots/g1_23dof_host/
```

All meshes but `head_link.STL` are identical to Unitree's G1 meshes.
The main task `Unitree-G1-23dof-GetUp-v0` uses the unitree_model USD and does not need them.
