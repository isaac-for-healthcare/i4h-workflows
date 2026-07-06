 
 
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

###################################################
######        teleop with xr device          ######
###################################################
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


###################################################
######           test without xr             ######
###################################################
python workflows/rheo/scripts/simulation/examples/test_spread_tablecloth.py \
  --viz kit --action_mode joint --robot h2 \
  --physics_backend physx --num_steps 1500

python workflows/rheo/scripts/simulation/examples/test_spread_tablecloth.py \
  --viz kit --action_mode teleop --robot h2 \
  --physics_backend newton --num_steps 1500


###################################################
######    reproduce CUDA-700 / CUDA-716       ######
###################################################
## Verifies the fix in commit 8e8f6860 "resolve CUDA700 by removing env.sim.reset".
## The --repro_cuda700 flag inserts:
##   render warmup (defers Newton's first CUDA-graph capture to first env.step)
##   -> env.sim.reset()  (the poison, under use_cuda_graph=True)
##   -> env.reset()
##   -> first env.step()  <-- crashes here
## Same bug surfaces as either
##   CUDA-700 (cudaErrorIllegalAddress) in narrow_phase/create_soft_contacts, or
##   CUDA-716 (cudaErrorMisalignedAddress) in wp_free_device_async
## depending on the newton/warp/driver combo. Without the flag the same command
## runs cleanly, which confirms env.sim.reset() is the sole trigger.
python -u workflows/rheo/scripts/simulation/examples/test_spread_tablecloth.py \
  --viz kit --action_mode teleop --robot h2 --physics_backend newton \
  --num_steps 20 --repro_cuda700 --repro_warmup 100 
echo "--- tail of /tmp/repro_cuda700.log ---"
grep -E "REPRO CUDA-700|CUDA error 7|STEP " /tmp/repro_cuda700.log | head -20
