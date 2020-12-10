# Generalization Guarantees for (Multi-Modal) Imitation Learning

[Paper](https://arxiv.org/abs/2008.01913) | [Review](https://drive.google.com/file/d/1VmLh07UuOVhDxGXh2YoVCJf3GvHNbG0M/view?usp=sharing) | [Experiment video](https://www.youtube.com/watch?v=dfXyHvOTolc&t=3s) | [5min presentation at CoRL 2020](https://www.youtube.com/watch?v=nabtvOWoIlo&feature=emb_logo)

[![Watch the video](https://img.youtube.com/vi/dfXyHvOTolc/maxresdefault.jpg)](https://www.youtube.com/watch?v=dfXyHvOTolc)

This repository includes codes for synthetic trainings of these robotic tasks in the paper:
1. Grasping diverse mugs
2. Planar box pushing using visual-feedback
3. Vision-based navigation through home environments

Although the codes for all examples are included here, only the pushing example can be run without any additional codes/resources. The other two examples require data from online object dataset and object post-processing, which can take significant amount of time to set up and involves licensing. Meanwhile, all objects (rectangular boxes) used for the pushing example can be generated through URDF files (`generativeBox.py`).

Moreover, we provide the pre-trained weights for the decoder network of the cVAE for the pushing example. The posterior policy distribution can be trained then using the weights and the prior distribution (unit Gaussians).

### Dependencies (`pip install` with python=3.7):
1. pybullet
2. torch
3. ray
4. cvxpy

### File naming:
- `..._bc.py` is for behavioral cloning training using collected demonstrations. 
- `..._es.py` is for PAC-Bayes ``fine-tuning'' using Natural Evolutionary Strategies. Also computes the final bound at the end of training.
- `..._bound.py` is for computing the final bound.

### Running the pushing example:
1. Generate a large number (3000) of boxes by running ```python generateBox.py --obj_folder=...``` and specifying the path to the object URDF files generated.
2. Modify ```obj_folder``` in `push_pac_easy.json` and `push_pac_hard.json`
3. Train pushing tasks ("Easy" or "Hard" difficulty) by running ```python trainPush_es.py push_pac_easy``` (or `hard`). The final bound is also computed by specifying `L` (number of policies sampled for each environment for computing the sample convergence bound) in the json file. (**Note:** the default number of training environments is 1000 as in the json files. With `num_cpus=20` on a moderately powerful desktop, it takes 20 minutes for each training step. We recommend training using Amazon AWS instance c5.24xlarge that has 96 threads. Also, a useful final bound requires large `L` and can take significant computations.)
4. Visualize pushing rollout by running ```python testPushRollout.py --obj_folder=... --posterior_path=...```. If `posterior_path` is not provided, the prior policy distribution (unit Gaussians) is used. Otherwise, the path should be `push_result/push_pac_easy/train_details` (or `hard`).

### Future release
1. Provide the mesh-processing code for mugs from ShapeNet.
2. Provide the collected demonstrations for both pushing and grasping examples.

(**Note:** we do not plan to release instructions to replicate results of the indoor navigation example in the near future. We plan to refine the simulation in a future version of the paper.)
