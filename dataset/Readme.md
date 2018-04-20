* SceneFlow includes three datasets: flything3d, driving and monkaa.
* You can train PSMNet with some of three datasets, or all of them.
* the following is the describtion of six subfolder.
```
# the disp folder of Driving dataset
driving_disparity  
# the image folder of Driving dataset
driving_frames_cleanpass

# the disp folder of  Flything3D dataset
frames_cleanpass  
# the image folder of  Flything3D dataset
frames_disparity  

# the disp folder of Monkaa dataset
monkaa_disparity  
# the image folder of Monkaa dataset
monkaa_frames_cleanpass
```
* Download the dataset from [this](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html). And unzip them to corresponding folder.

* `data_scene_flow_2015` is the folder for kitti15. You can unzip kitti15 to this folder. This will be used in **test** pahse.
