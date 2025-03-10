

If your system GLIBC doesn't match, and you get this error then run
```
conda install -c conda-forge gcc=13.3.0
```

Make sure the following pip packages are available:
```
pip install pyrealsense2
pip install rti.connext
pip install holoscan
```

To run realsense camera
```
python3 holoscan/realsense/camera.py
```
