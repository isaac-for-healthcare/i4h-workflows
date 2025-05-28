import distutils
import os

import ntplib


def get_ntp_offset(
    ntp_server=os.environ.get("NTP_SERVER_HOST", "pool.ntp.org"),
    ntp_port=int(os.environ.get("NTP_SERVER_PORT", "123")),
    version=3,
):
    client = ntplib.NTPClient()
    try:
        response = client.request(ntp_server, port=ntp_port, version=version)
        print(f"NTP Sync offset time: {response.offset:.6f}")
        return response.offset
    except Exception as e:
        print(f"[WARN] NTP sync failed: {e}")
        return 0


def strtobool(s):
    return False if s is None else s if isinstance(s, bool) else bool(distutils.util.strtobool(s))
