# Original work
This repo is implementation for poinet.pytorch.
Original work: https://github.com/fxia22/pointnet.pytorch
# PointNet.pytorch
This repo is implementation for PointNet(https://arxiv.org/abs/1612.00593) in pytorch. The model is in `pointnet/model.py`.

It is tested with Python 3.9.21, Pytorch 2.5.1, CUDA 11.8.

# Download data and running

```
git clone https://github.com/rinisme00/pointnet.pytorch-modified
cd pointnet.pytorch
pip install -e .
```

Download and build visualization tool
```
cd scripts
bash build.sh #build C++ code for visualization
bash download.sh #download dataset
```

Training 
```
cd utils
python train_classification.py --dataset <dataset path> --nepoch=<number epochs> --num_classes <number of classes> --outf <output path>
python train_segmentation.py --dataset <dataset path> --nepoch=<number epochs> 
```

Use `--feature_transform` to use feature transform.

# Convert Dataset
```
cd Convert_PCD
```
Put the .pcd files in the /in folder. The ```convert.py``` will convert them into .pts (the coordinates of point clouds in 3D space) and .seg (labels for each of point clouds) format.
```
python convert.py
```

# Visualization
```
mkdir results
python show_seg.py --model <the path of .pth files after training> --dataset <dataset path> --split test --idx=<depends on the number of classes of the object> --save results/<image.png> --export results/<.pth model> --export_gt results/<ground truth model> --export_edges results/<.pth model> --conf_thresh <threshold for unassigned>
```

# Example
In this section, I tested with brick object. Here is the visualization of the object after segmentation:
![image](https://github.com/user-attachments/assets/64682ad9-4d77-4cd5-ba6f-7978718efb09)
![image](https://github.com/user-attachments/assets/8ca53a8d-58d9-4187-b455-865bd54f2cab)
![image](https://github.com/user-attachments/assets/d7428006-bc90-4d04-ad83-d4c849c7b784)
![image](https://github.com/user-attachments/assets/904947a9-5114-44b5-a260-ff5557b1ab9c)
![image](https://github.com/user-attachments/assets/4e885568-e960-473c-892b-8db196f11df2)
![image](https://github.com/user-attachments/assets/5caca3d2-8481-40e5-b693-a0b71f14094c)
![image](https://github.com/user-attachments/assets/e8742802-f1cb-47c0-a5b4-b6fa4a10b6a9)
