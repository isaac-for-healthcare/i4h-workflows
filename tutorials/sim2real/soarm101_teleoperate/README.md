### Teleoperate

```bash
# Terminal 1: Hardware driver with real-time mode
python3 host_soarm_driver.py --port 9999 --realtime

# Terminal 2: Isaac Sim with real-time mode
# need to surce your built isaac sim env first
source /home/venn/Desktop/code/isaacsim/_build/linux-x86_64/release/setup_conda_env.sh
python isaac_sim_source.py --port 9999 --realtime
```
