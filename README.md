Assignment 4
===================================

##  0. Setup

Assignment 4 will build on top of the assignment 3 codebase. Copy the following files and directories into your assignment 3 codebase:

```bash
a4/
render_functions.py
data_utils.py
```

##  1. Sphere Tracing (30pts)

In this part you will implement sphere tracing for rendering an SDF, and use this implementation to render a simple torus. You will need to implement the `sphere_tracing` function in `a4/renderer.py`. This function should return two outpus: (`points, mask`), where the `points` Tensor indicates the intersection point for each ray with the surface, and `masks` is a boolean Tensor indicating which rays intersected the surface.

You can run the code for part 1 with:
```bash
# mkdir images (uncomment when running for the first time)
python -m a4.main --config-name=torus
```

This should save `part_1.gif` in the `images' folder. Please include this in your submission along with a short writeup describing your implementation.

![Torus](images/part_1.gif)

##  2. Optimizing a Neural SDF (30pts)

In this part, you will implement an MLP architecture for a neural SDF, and train this neural SDF on point cloud data. You will do this by training the network to output a zero value at the observed points. To encourage the network to learn an SDF instead of an arbitrary function, we will use an 'eikonal' regularization which enforces the gradients of the predictions to behave in a certain way (search lecture slides for hints).

In this part you need to:

* **Implement a MLP to predict distance**: You should populate the `NeuralSurface` class in `a4/implicit.py`. For this part, you need to define a MLP that helps you predict a distance for any input point. More concretely, you would need to define some MLP(s) in  `__init__` function, and use these to implement the `get_distance` function for this class. Hint: you can use a similar MLP to what you used to predict density in Assignment 3, but remember that density and distance have different possible ranges!

* **Implement Eikonal Constraint as a Loss**: Define the `eikonal_loss` in `a4/losses.py`.

After this, you should be able to train a NeuralSurface representation by:
```bash
python -m a4.main --config-name=points
```

This should save save `part_2_input.gif` and `part_2.gif` in the `images` folder. The former visualizes the input point cloud used for training, and the latter shows your prediction which you should include on the webpage alongwith brief descriptions of your MLP and eikonal loss. You might need to tune hyperparameters (e.g. number of layers, epochs, weight of regularization, etc.) for good results.

![Bunny geometry](images/part_2.gif)

##  3. VolSDF (30 pts)


In this part, you will implement a function converting SDF -> volume density and extend the `NeuralSurface` class to predict color. 

* **Color Prediction**: Extend the the `NeuralSurface` class to predict per-point color. You may need to define a new MLP (a just a few new layers depending on how you implemented Q2). You should then implement the `get_color` and `get_distance_color` functions.

* **SDF to Density**: Read section 3.1 of the [VolSDF Paper](https://arxiv.org/pdf/2106.12052.pdf) and implement their formula converting signed distance to density in the `sdf_to_density` function in `a4/renderer.py`. In your write-up, give an intuitive explanation of what the parameters `alpha` and `beta` are doing here. Also, answer the following questions:
1. How does high `beta` bias your learned SDF? What about low `beta`?
2. Would an SDF be easier to train with volume rendering and low `beta` or high `beta`? Why?
3. Would you be more likely to learn an accurate surface with high `beta` or low `beta`? Why?

After implementing these, train an SDF on the lego bulldozer model with

```bash
python -m a4.main --config-name=volsdf
```

This will save `part_3_geometry.gif` and `part_3.gif`. Experiment with hyper-parameters to and attach your best results on your webpage. Comment on the settings you chose, and why they seem to work well.

![Bulldozer geometry](images/part_3_geometry.gif) ![Bulldozer color](images/part_3.gif)


## 4. Neural Surface Extras (CHOOSE ONE! More than one is extra credit)

### 4.1. Render a Large Scene with Sphere Tracing (10 pts)
In Q1, you rendered a (lonely) Torus, but to the power of Sphere Tracing lies in the fact that it can render complex scenes efficiently. To observe this, try defining a ‘scene’ with many (> 20) primitives (e.g. Sphere, Torus, or another SDF from [this website](https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm) at different locations). See Lecture 2 for equations of what the ‘composed’ SDF of primitives is. You can then define a new class in `implicit.py` that instantiates a complex scene with many primitives, and modify the code for Q1 to render this scene instead of a simple torus.
### 4.2 Fewer Training Views (10 pts)
In Q3, we relied on 100 training views for a single scene. A benefit of using Surface representations, however, is that the geometry is better regularized and can in principle be inferred from fewer views. Experiment with using fewer training views (say 20) -- you can do this by changing [train_idx in data laoder](https://github.com/learning3d/assignment3/blob/main/dataset.py#L123) to use a smaller random subset of indices). You should also compare the VolSDF solution to a NeRF solution learned using similar views.
### 4.3 Alternate SDF to Density Conversions (10 pts)
In Q3, we used the equations from [VolSDF Paper](https://arxiv.org/pdf/2106.12052.pdf) to convert SDF to density. You should try and compare alternate ways of doing this e.g. the ‘naive’ solution from the [NeuS paper](https://arxiv.org/pdf/2106.10689.pdf), or any other ways that you might want to propose!
