## Project Conventions

- Each paper implementation lives under its taxonomy category directory, named `YYYY_PaperName/`.
- Categories: `Rendering_and_Representation/`, `Geometry_and_Structure/`, `Generative_3D/`, `Perception/`, `6DoF_Pose/`.
- Category 4 (Perception) has sub-categories: 4-1 Point Cloud Analysis, 4-2 Scene Graphs.
- Category 6 (Physical AI) has sub-categories: 6-1 Dynamic 4D, 6-2 Physics-based, 6-3 Affordance.
- To decide where a paper belongs, ask: "What is the PRIMARY optimization objective?"
- Every new project must include its own `README.md` with setup instructions, usage, and BibTeX citation.
- Use `/paper-to-note` to create a `study_note.md` for each studied paper.
- The standardized stack is PyTorch 2.7.0 + CUDA 12.8. Prefer mamba for environment creation.
- The root `README.md` is in Korean — preserve this convention when updating the taxonomy.
