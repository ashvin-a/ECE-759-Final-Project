# OpenSim
An open source simulator that is hardware agnostic and can be used for simulation for robot learning and other learning algorithms This is the plan - we will build a simulator.
For that, we will execute it in three steps by building the three main components of a simulator:

1. The Compute Engine - Here, we plan to use Ray and Taichi Lang (randum Python aan 😌) for building the 
2. The Physics Engine
3. The Renderer 
# Resources
1. [Ray](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html) => For high performance computing using Python
2. [Taichi Lang](https://www.taichi-lang.org/) => For seamless working in both GPU and CPU, making the system truly hardware agnostic
3. [Wgpu-py](https://github.com/pygfx/wgpu-py) => For rendering the graphics. 

Thanlkaalam ithrem nokkam. Now, nammade agenda ingane aan(Gemini ser 🛐).**(SUBJECT TO CHANGE)**

## Phase 1: The "Hello World" of HPC (Weeks 1-2)
Goal: Get a kernel running on the GPU without crashing.
Focus: Taichi Syntax & Data-Oriented Design.
- Week 1: Taichi Basics
  - Learn: ti.init(arch=ti.gpu), ti.field, and @ti.kernel. Understand why for i in range(10000) inside a kernel runs in parallel, but outside it runs serially.
  - Task: Create a 10,000 particle system. Spawn 10k particles at random positions. Make them fall under gravity ($v = v + g \cdot dt$). Reset them when they hit $y=0$.
  - Visual Check: Just use ti.GUI (Taichi's built-in simple debug viewer) for now. Do not touch wgpu-py yet.
- Week 2: The Data Structure (Struct of Arrays)
  - Learn: Why class Particle: pos, vel (AoS) destroys cache locality.
  - Task: Refactor your code to use ti.dataclass.
  - Math: Review Quaternions. You need them for rotation. A rotation matrix is too heavy (9 floats vs 4 floats).
  - Deliverable: A simulation of 1,000 tumbling cubes (rendering them as points is fine for now) falling and hitting the floor.


## Phase 2: The Physics Engine (Weeks 3-8)
Goal: Implement XPBD (Extended Position Based Dynamics).
Focus: The "Solver" Loop.
- Week 3: XPBD Integrator
  - Learn: The XPBD algorithm (Matthias Müller).
  - Task: Implement the "Sub-stepping" loop.predict_position = current_pos + velocity * dtsolve_constraints(predict_position)update_velocity
  - Deliverable: A single cube that you can drag around with the mouse (using ti.GUI interactions) that feels "heavy" and conserved momentum.
- Week 4-5: Collision Detection (The Hard Part)
  - Learn: Broad Phase vs. Narrow Phase.
  - Task (Broad Phase): Implement Spatial Hashing. Map every object to a grid cell ID. Only check collisions between objects in the same cell.
  - Task (Narrow Phase): Implement SDF (Signed Distance Field) collisions for Box-Box or Sphere-Box.
  - Deliverable: Drop 2,000 cubes into a pile. They should stack, not explode or pass through each other.

- Week 6: Joints & Articulation
  - Learn: Lagrange Multipliers (conceptually) vs. XPBD Compliance.
  - Task: Create a "Revolute Joint" (Hinge).
    - Constraint: Point A on Body 1 must be at the same location as Point B on Body 2.
    - Constraint: The rotation axes must align.
  - Deliverable: A "chain" of 5 cubes linked together swinging like a pendulum.

- Week 7-8: The Robot Loader (MJCF)
  - Learn: XML parsing in Python.
  - Task: Write a parser for standard MuJoCo MJCF files.
  - Deliverable: Load a simple "Ant" or "Humanoid" robot into your simulation. It will look like a ragdoll collapsing, but it should load.

## Phase 3: The Renderer (Weeks 9-11)
Goal: High-fidelity visualization using wgpu-py.
Focus: Graphics Pipeline & Zero-Copy.
- Week 9: WGPU Basics
  - Learn: Vertex Buffers, Index Buffers, and WGSL Shaders.
  - Task: Render a single static triangle using wgpu-py. (This is harder than it sounds if you've never done graphics).
- Week 10: The Bridge (Zero-Copy)
  - Learn: ti.ndarray and Vulkan/Metal interoperability.
  - Task: Map the Taichi particle positions directly to the WGPU Vertex Buffer.
  - Deliverable: Your 10,000 cube pile simulation, but now rendered with lighting and shadows in WGPU.
- Week 11: Instanced RenderingLearn: How to draw 10,000 meshes with 1 draw call.
  - Task: Implement "Hardware Instancing" in your shader.
  - Deliverable: Real-time 60FPS visualization of the simulation.
  
## Phase 4: The RL Wrapper (Weeks 12-14) (We may not be doing this since its beyond scope of HPC. Baaki set aakan nokkam)
Goal: Make it "Learnable."
Focus: Python API Design.
- Week 12: The Gym Interface
  - Task: Wrap your entire engine in a class MySimEnv.Implement step(action), reset(), and get_obs().
  - Crucial: Ensure step() takes a batch of actions (e.g., actions for 4096 robots at once).
- Week 13: Sim-to-Real Basics
  - Task: Add "Domain Randomization."Randomly change the friction and mass of the robot every reset(). This makes the trained policy robust.
- Week 14: Final Demo
  - Task: Train a robot to walk. Connect your env to a standard PPO implementation (like CleanRL or RSL_RL).
  - Deliverable: A video of 4,096 ants learning to walk simultaneously on your GPU.
