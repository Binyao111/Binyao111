<p align="center">
  <img src="assets/banner.png" alt="Binyao Guo — Computational Physics · Electromagnetic Surrogates · Neural Operators" width="100%"/>
</p>

<p align="center">
  <a href="https://github.com/Binyao111">
    <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=20&pause=1200&color=00B4DB&center=true&vCenter=true&width=720&lines=Neural+operators+for+PDEs+%26+path-dependent+materials;Discretization-consistent+spectral+autoencoders;Electromagnetic+surrogates+from+ANSYS+HFSS;Differentiable+FEM+%2F+meshfree+solvers+in+JAX" alt="Neural operators for PDEs and path-dependent materials · Discretization-consistent spectral autoencoders · Electromagnetic surrogates from ANSYS HFSS · Differentiable FEM / meshfree solvers in JAX"/>
  </a>
</p>

<p align="center">
  <a href="mailto:guo00544@umn.edu"><img src="https://img.shields.io/badge/Email-guo00544%40umn.edu-D14836?style=flat-square&logo=gmail&logoColor=white" alt="Email"/></a>
  <a href="https://binyaoguo.wordpress.com/"><img src="https://img.shields.io/badge/Website-binyaoguo.wordpress.com-21759B?style=flat-square&logo=wordpress&logoColor=white" alt="Website"/></a>
  <a href="https://github.com/IntelligentMechanicsLab"><img src="https://img.shields.io/badge/Lab-IntelligentMechanicsLab-181717?style=flat-square&logo=github&logoColor=white" alt="Intelligent Mechanics Lab"/></a>
  <img src="https://img.shields.io/badge/Minneapolis%2C%20MN-University%20of%20Minnesota-7A0019?style=flat-square" alt="University of Minnesota"/>
</p>

---

### 👋 About me

- 🎓 **Ph.D. student in Civil Engineering** at the **University of Minnesota** (advisor: Prof. QiZhi He), also pursuing an **M.S. in Computer Science** at UMN.
- 🧠 I build **neural operators and latent surrogates** for physical simulation, from path-dependent plasticity to electromagnetic waveguides and phase-field PDEs.
- 🔧 I couple learned models with **differentiable solvers** (JAX-FEM, RKPM, neural-integrated meshfree) so they can run inside real simulations.
- 🏫 Previously: M.S. Structural Engineering @ **UIUC**; B.Eng. Civil Engineering (dual degree) @ **University of Leeds** (First-Class) & **Southwest Jiaotong University**.
- 🏆 UMN Data Science Initiative ADC Graduate Fellowship (2025) · DOE Genesis Mission Hackathon @ Argonne National Laboratory (2026).

---

### 🔬 Research highlights

<table>
  <tr>
    <td width="50%" valign="top">
      <h4>⚡ Electromagnetic Waveguide Surrogate</h4>
      <img src="assets/em_waveguide.png" alt="ANSYS HFSS electric-field maps of empty and dielectric-loaded waveguides, and HFSS vs. autoencoder surrogate prediction" width="100%"/>
      <sub>Convolutional autoencoder + frequency→latent mapper trained on <b>ANSYS HFSS</b> field data. With only <b>6 latent variables</b>, interpolation error is <b>1.86%</b> (empty) and <b>4.33%</b> (dielectric-loaded) for cylindrical waveguides.</sub>
    </td>
    <td width="50%" valign="top">
      <h4>🌊 SANO: Spectral Autoencoding Neural Operator</h4>
      <img src="assets/sano_cahn_hilliard.gif" alt="Animated Cahn–Hilliard phase-field evolution: ground truth vs. SANO autoregressive latent rollout" width="100%"/>
      <img src="assets/sano_cross_resolution.png" alt="Zero-shot SANO decoding of one latent vector on 32², 64², 128² and 256² grids" width="100%"/>
      <sub>Discretization-consistent nonlinear reduced representations. <b>Top:</b> autoregressive Cahn–Hilliard rollout in latent space. <b>Bottom:</b> one latent code decoded zero-shot on <b>32²–256²</b> grids. Wave-equation trajectory error <b>0.4353 → 0.1914</b> vs. RONOM.</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h4>🧱 HANO: History-Aware Neural Operator</h4>
      <img src="assets/hano.png" alt="HANO architecture (Fourier and attention-enhanced U-Net/Fourier layers) and stress–strain loops predicted from mid-history starts by HANO vs. a recurrent baseline" width="100%"/>
      <sub>Autoregressive U-Net + Fourier neural operator + attention for path-dependent plasticity. Starting predictions mid-history (red dot), HANO tracks the hysteresis loop while a recurrent baseline drifts. Stress NRMSE <b>2.3% → 0.6%</b>. First author, <i>CMAME</i> 2025 · <a href="https://github.com/IntelligentMechanicsLab/HANO_inelasticity">code</a>.</sub>
    </td>
    <td width="50%" valign="top">
      <h4>🔁 Learned Plasticity inside JAX-FEM</h4>
      <img src="assets/hano_jaxfem.png" alt="Plate with a hole: reference vs. HANO-coupled JAX-FEM stress fields (σxx, σyy) and absolute error" width="100%"/>
      <sub>HANO embedded as the constitutive model in nonlinear <b>JAX-FEM</b> solves (Newton, Jacobian-vector products, GMRES, line search). Shown: plate-with-hole stresses. In a cyclic J2-plasticity test it converged over <b>18 load increments</b> with final errors of <b>2.90%</b> stress, <b>1.05%</b> displacement, and <b>0.96%</b> reaction force. <i>(NSF project)</i></sub>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h4>🪨 DOE Genesis: Pore-Network Dissolution</h4>
      <img src="assets/genesis_clusters.png" alt="Grid of pore-network dissolution patterns clustered into five groups (G1–G5) using PCA and shape features" width="100%"/>
      <sub>AI-guided critical-mineral extraction. 80 PCA coefficients + 5 morphology features with K-means (<b>K = 5</b>) organize <b>1,881</b> dissolution simulations into pattern families, from compact and uniform dissolution to wormholes and ramified branches.</sub>
    </td>
    <td width="50%" valign="top">
      <h4>🏗️ Neural Topology Optimization (NTO-PCO)</h4>
      <img src="assets/topology_optimization.png" alt="Neural topology optimization framework (CNN reparameterization + PyGRANSO), optimized MBB beam, bridge and L-shape designs, and a multi-material bridge" width="100%"/>
      <sub>CNN reparameterization of the design field, solved with principled constrained optimization (<b>PyGRANSO</b>) under physics, volume, and combinatorial constraints. Extends to <b>multi-material</b> designs. Co-author.</sub>
    </td>
  </tr>
</table>

<details>
<summary><b>More projects</b></summary>
<br/>

- 🕸️ **Differentiable Neural-Integrated Meshfree (NIM)** method for finite-strain hyperelasticity, forward and inverse (2nd author, <i>Engineering with Computers</i> 2025).
- 🧮 **Differentiable RKPM solver** in JAX, developed alongside the HANO coupling work.
- 🚀 **NSRI: generative Bayesian learning for hypersonic flowfields.** A latent diffusion model generates 2D hypersonic cylinder flowfields as a learned prior for sparse-observation inference.
- 🏢 **UIUC:** point-cloud quality assessment for 3D-printed concrete; U-Net crack segmentation + structure-from-motion for 2D-to-3D crack localization.

</details>

---

### 📄 Selected publications

- **B. Guo**, Z. Lin, Q. He. *History-Aware Neural Operator: Robust Data-Driven Constitutive Modeling of Path-Dependent Materials.* **Computer Methods in Applied Mechanics and Engineering** 447 (2025). [[DOI]](https://doi.org/10.1016/j.cma.2025.118358) [[arXiv]](https://arxiv.org/abs/2506.10352) [[code]](https://github.com/IntelligentMechanicsLab/HANO_inelasticity)
- H. Du, **B. Guo**, Q. He. *Differentiable Neural-Integrated Meshfree Method for Forward and Inverse Modeling of Finite Strain Hyperelasticity.* **Engineering with Computers** 41, 1597–1617 (2025). [[DOI]](https://doi.org/10.1007/s00366-024-02090-z) [[arXiv]](https://arxiv.org/abs/2407.11183)
- **B. Guo**, Q. He. *Spectral Autoencoding Neural Operator (SANO): Discretization-Consistent Nonlinear Reduced Representations of Scientific Fields.* Manuscript.
- R. Devera, **B. Guo**, B. Liang, Q. He, J. Sun. *Neural Topology Optimization with Principled Constrained Optimization.* Manuscript.
- S. Dietterich, **B. Guo**, S. Hanany, Q. He. *Advancing Data-Driven Electromagnetics Simulations with the Convolutional Auto-Encoder.* In preparation.
- Y. Zhao, **B. Guo**, I. Aziz, M. Alipour. *An Optimization Framework for UAS-Based Infrastructure Inspection Path Planning.* **Computing in Civil Engineering 2023**, 890–898.
- Y. Zhao, **B. Guo**, H. Che, M. Alipour. *Developing Human Sensing Platforms for Digitizing Visual Inspections of Critical Infrastructure.* **Structures Congress 2023**, 340–353.

---

### 🛠️ Toolbox

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/JAX-5E97F6?style=for-the-badge" alt="JAX"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/MATLAB-0076A8?style=for-the-badge&logo=mathworks&logoColor=white" alt="MATLAB"/>
  <img src="https://img.shields.io/badge/ANSYS%20HFSS-FFB71B?style=for-the-badge&logo=ansys&logoColor=black" alt="ANSYS HFSS"/>
  <img src="https://img.shields.io/badge/Abaqus-005386?style=for-the-badge&logo=dassaultsystemes&logoColor=white" alt="Abaqus"/>
  <img src="https://img.shields.io/badge/FEM%20%2F%20RKPM-2C5364?style=for-the-badge" alt="FEM / RKPM"/>
</p>
