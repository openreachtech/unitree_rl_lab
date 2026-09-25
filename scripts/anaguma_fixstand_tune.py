#!/usr/bin/env python3
"""Anaguma の FixStand(起立) ゲインを mujoco で実測して決める。

`deploy/robots/anaguma/config/config.yaml` の FixStand.kp/kd/ts と同じ PD ランプを
python で再現し、3秒後の 胴体高 / 傾き / ピークトルク / 飽和率 / 姿勢誤差 を出す。
目で決めると「プルプル震えながらなんとか立つ」状態を掴めないので、
**飽和率(トルクが ctrlrange に張り付いている時間の割合)** を必ず見る。

    /home/tanaka/isaacsim/anaguma/venv/bin/python scripts/anaguma_fixstand_tune.py
"""
import os; os.environ.setdefault("MUJOCO_GL","egl")
import numpy as np, mujoco, yaml

SCENE="/home/tanaka/isaacsim/unitree_mujoco/unitree_robots/anaguma/scene_flat.xml"
DEP="/home/tanaka/isaacsim/unitree_rl_lab/logs/rsl_rl/anaguma_longjump_v1/params/deploy.yaml"
SDK=["FR_hip_joint","FR_thigh_joint","FR_calf_joint","FL_hip_joint","FL_thigh_joint","FL_calf_joint",
     "RR_hip_joint","RR_thigh_joint","RR_calf_joint","RL_hip_joint","RL_thigh_joint","RL_calf_joint"]

m=mujoco.MjModel.from_xml_path(SCENE); d=mujoco.MjData(m)
jid=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in SDK]
qadr=np.array([m.jnt_qposadr[i] for i in jid]); vadr=np.array([m.jnt_dofadr[i] for i in jid])
act=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n.replace("_joint","")) for n in SDK])
tau_lim=np.abs(m.actuator_ctrlrange[act]).max(axis=1)
feet=[i for i in range(m.ngeom)
      if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or "").find("foot")>=0]
base=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"base")

cfg=yaml.safe_load(open(DEP))
# deploy.yaml の default_joint_pos は policy 順。config.yaml の qs は SDK 順。
# 変換は sdk[joint_ids_map[i]] = policy[i] (逆向きにすると無意味な姿勢になる)。
stand=np.zeros(12); dpol=np.array(cfg["default_joint_pos"])
for pol_i,sdk_i in enumerate(cfg["joint_ids_map"]): stand[sdk_i]=dpol[pol_i]

def sim(kp,kd,ramp=2.5,hold=1.5):
    mujoco.mj_resetData(m,d); d.qpos[:]=0; d.qpos[3]=1.0
    mujoco.mj_forward(m,d)
    d.qpos[2]=m.geom_size[feet[0]][0]-min(d.geom_xpos[f][2] for f in feet)+0.002
    mujoco.mj_forward(m,d)
    q0=d.qpos[qadr].copy(); t=0.0; peak=0.0; sat=0; n=0
    while t<ramp+hold:
        a=min(t/ramp,1.0); tgt=q0*(1-a)+stand*a
        tau=kp*(tgt-d.qpos[qadr])-kd*d.qvel[vadr]
        peak=max(peak,np.abs(tau).max())
        sat+=int(np.any(np.abs(tau)>=tau_lim*0.999)); n+=1
        d.ctrl[act]=np.clip(tau,-tau_lim,tau_lim); mujoco.mj_step(m,d); t+=m.opt.timestep
    R=d.xmat[base].reshape(3,3)
    tilt=np.degrees(np.arccos(np.clip(R[2,2],-1,1)))
    err=np.sqrt(np.mean((d.qpos[qadr]-stand)**2))
    return d.qpos[2],tilt,peak,100.0*sat/n,err

if __name__=="__main__":
    print(f"立ち姿勢(SDK順) = {np.round(stand,4)}")
    print(f"トルク上限 = {tau_lim[:3]} N*m")
    mujoco.mj_resetData(m,d); mujoco.mj_forward(m,d)
    print(f"[MJCF既定 spawn] base z={d.qpos[2]:.3f} 最下足 z={min(d.geom_xpos[f][2] for f in feet):.4f}"
          " (負なら床に埋まっている)")
    print(f"\n{'kp':>5}{'kd':>5}{'胴体高':>9}{'傾き':>8}{'ピークτ':>10}{'飽和%':>8}{'姿勢誤差':>10}")
    for kp in (60,80,100,120,150,200,250):
        for kd in (2.0,4.0,6.0,8.0,10.0):
            z,tl,pk,sa,er=sim(kp,kd)
            print(f"{kp:>5}{kd:>5.1f}{z:>9.3f}{tl:>8.1f}{pk:>10.1f}{sa:>8.1f}{er:>10.4f}")
