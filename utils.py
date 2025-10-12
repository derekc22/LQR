import mujoco
import numpy as np

def load_model(model_path: str) -> tuple[mujoco.MjModel, mujoco.MjData]:    
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    return m, d

def reset_stoch(m: mujoco.MjModel, 
                d: mujoco.MjData, 
                keyframe: str) -> None:
    init_qpos = m.keyframe(keyframe).qpos
    init_qvel = m.keyframe(keyframe).qvel
    mujoco.mj_resetData(m, d) 
    d.qpos = np.random.uniform(-1.5, 1.5) * init_qpos
    d.qvel = init_qvel
    mujoco.mj_forward(m, d)
    

def get_body_id(m: mujoco.MjModel,
                body: str) -> int:
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)


def get_body_mass(m: mujoco.MjModel,
                  body: str) -> int:
    return m.body_mass[get_body_id(m, body)]


def get_body_size(m: mujoco.MjModel, 
                  body: str) -> np.ndarray:
    geom_sizes = m.geom_size[m.geom_bodyid == get_body_id(m, body)]
    return geom_sizes[0]
    # The shape of geom_sizes is always (N, 3) - where N is the number of geoms for that body
    # We return geom_sizes[0] to return the geom_size array associated with the first chronological geom


def get_gravity(m: mujoco.MjModel) -> int:
    return m.opt.gravity[-1]


def get_body_inertia(m: mujoco.MjModel,
                     body: str) -> int:
    return m.body_inertia[get_body_id(m, body)]

