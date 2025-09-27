import mujoco

def load_model(model_path: str) -> tuple[mujoco.MjModel, mujoco.MjData]:    
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    return m, d

def reset(m: mujoco.MjModel, 
          d: mujoco.MjData, 
          keyframe: str) -> None:
    init_qpos = m.keyframe(keyframe).qpos
    init_qvel = m.keyframe(keyframe).qvel
    mujoco.mj_resetData(m, d) 
    d.qpos = init_qpos
    d.qvel = init_qvel
    mujoco.mj_forward(m, d)