# Inventory-Management:

1. Download Anaconda for python 3.6:
Download link: https://www.anaconda.com/download/
  -Add to enviorment variable path: C:\Program Files (x86)\Anaconda3\Scripts

3. Download Cuda and Cudnn:
https://www.youtube.com/watch?v=Ebo8BklTtmc
  -Add Cuda bin to enviorment variable path: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin
  -Add Cuda libnvvp to enviorment variable path: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp
  -Extract Cudnn Folder
  -Move CUDA for Cudnn folder to ProgramFiles
  -Add Cudnn to to enviorment variable path: C:\cuda\bin
  - run this too: conda install -c anaconda msgpack-python
 
4. Install Tensorflow GPU:
conda create -n tensorflow
activate tensorflow
pip install --ignore-installed --upgrade tensorflow-gpu 

4. Download Object Detection API:
Tutorial: https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/
More Videos: https://www.youtube.com/watch?v=COlbP62-B-U&list=PLQVvvaa0QuDcNK5GeCQnxYnSSaar2tpku

5. Download Code from Github into: models/research/object_detection
