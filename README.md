# V-Touching-Data
This repository is dedicated to storing data for V-Touching.

## Usage
1. First, launch Unity and open the project directory `TouchMaterialServer`. The project interface will be displayed in Unity.

2. Drag the directory `resylt_Database` into the "Project" column:
   <div align="center">
       <img src="Data/usage/drag_postion.png" alt="Figure 1" />
   </div>

3. Create a *Material* for the desired data within the `Data` directory. Let's take **notebook_cover** as an example. Right-click on **notebook_cover**, select ```Create->Material```, and give it a custom name (let's name it **notebook_over_mat**):
   <div align="center">
       <img src="Data/usage/create_material.png" alt="Figure 2" />
   </div>

4. Left-click on **notebook_over_mat** to display its information in the "Inspector" column. Then, select **notebook_over** and drag it into the left rectangle box of the "Albedo" text:
   <div align="center">
       <img src="Data/usage/Albedo.png" alt="Figure 3" />
   </div>

5. Right-click on "App" in the "Hierarchy" column and select ```3D Object->Cube``` to create a *Cube*:
   <div align="center">
       <img src="Data/usage/create_cube.png" alt="Figure 4" />
   </div>

6. Left-click on the newly created *Cube*, click on **Add Component** at the bottom of the "Inspector" column, scroll down, then find and click on ```Scripts->TouchMaterial.Server->Tactile Object.```

1. Finally, drag **notebook_over_mat** and **notebook_over_acc** to the positions indicated by the arrows in the following image:
   <div align="center">
       <img src="Data/usage/final.png" alt="Figure 4" />
   </div>

2. You're done!
