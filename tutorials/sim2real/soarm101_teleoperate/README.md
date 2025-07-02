### Teleoperate

```bash
# Terminal 1: Hardware driver with real-time mode
python3 host_soarm_driver.py --port 9999 --realtime

# Terminal 2: Isaac Sim with real-time mode
python isaac_sim_source.py --port 9999 --realtime
```
