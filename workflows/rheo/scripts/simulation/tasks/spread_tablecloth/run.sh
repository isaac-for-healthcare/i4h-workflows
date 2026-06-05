 
 
uv venv --python 3.12 --seed env_isaaclab
source env_isaaclab/bin/activate
uv pip install --upgrade pip 

## install isaacsim
uv pip install --pre isaacsim[all,extscache]==6.0.0rc53+release.41093.1985cf6c.gl \
  --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/sw-isaacsim-pypi/simple --extra-index-url https://pypi.nvidia.com/simple

## install isaaclab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git switch develop
./isaaclab.sh --install


### terminal 1
echo "NV_CXR_ENABLE_PUSH_DEVICES=0" > ~/handtracking.env
python -m isaacteleop.cloudxr --cloudxr-env-config=~/handtracking.env

# termina 2


## run source /path/to/.cloudxr/run/cloudxr.env shown in terminal 1 first
## change the cmd to your own path
source /home/mxgu/.cloudxr/run/cloudxr.env

## run the following command in terminal 2
python workflows/rheo/scripts/simulation/record_demos_tablecloth.py \
  --task Isaac-Spread-Tablecloth-G129-Inspire-Teleop \
  --num_demos 10 \
  --dataset_file datasets/tablecloth/demo.hdf5 \
  --device cuda:0 --enable_pinocchio --viz kit --xr \
  --enable_cameras --no-auto_launch_cloudxr