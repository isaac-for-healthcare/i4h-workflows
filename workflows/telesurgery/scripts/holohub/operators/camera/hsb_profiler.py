import logging
import math
import os

import cupy as cp
import holoscan
from cuda import cuda

import hololink as hololink_module

MS_PER_SEC = 1000.0
US_PER_SEC = 1000.0 * MS_PER_SEC
NS_PER_SEC = 1000.0 * US_PER_SEC
SEC_PER_NS = 1.0 / NS_PER_SEC


def get_timestamp(metadata, name):
    s = metadata[f"{name}_s"]
    f = metadata[f"{name}_ns"]
    f *= SEC_PER_NS
    return s + f


def record_times(recorder_queue, metadata):
    #
    now = datetime.datetime.utcnow()
    #
    frame_number = metadata.get("frame_number", 0)

    # frame_start_s is the time that the first data arrived at the FPGA;
    # the network receiver calls this "timestamp".
    frame_start_s = get_timestamp(metadata, "timestamp")

    # After the FPGA sends the last sensor data packet for a frame, it follows
    # that with a 128-byte metadata packet.  This timestamp (which the network
    # receiver calls "metadata") is the time at which the FPGA sends that
    # packet; so it's the time immediately after the the last byte of sensor
    # data in this window.  The difference between frame_start_s and frame_end_s
    # is how long it took for the sensor to produce enough data for a complete
    # frame.
    frame_end_s = get_timestamp(metadata, "metadata")

    # received_timestamp_s is the host time after the background thread woke up
    # with the nofication that a frame of data was available.  This shows how long
    # it took for the CPU to actually run the backtground user-mode thread where it observes
    # the end-of-frame.  This background thread sets a flag that will wake up
    # the pipeline network receiver operator.
    received_timestamp_s = get_timestamp(metadata, "received")

    # operator_timestamp_s is the time when the next pipeline element woke up--
    # the next operator after the network receiver.  This is used to compute
    # how much time overhead is required for the pipeline to actually receive
    # sensor data.
    operator_timestamp_s = get_timestamp(metadata, "operator_timestamp")

    # complete_timestamp_s is the time when visualization finished.
    complete_timestamp_s = get_timestamp(metadata, "complete_timestamp")

    recorder_queue.append(
        (
            now,
            frame_start_s,
            frame_end_s,
            received_timestamp_s,
            operator_timestamp_s,
            complete_timestamp_s,
            frame_number,
        )
    )


def save_timestamp(metadata, name, timestamp):
    # This method works around the fact that we can't store
    # datetime objects in metadata.
    f, s = math.modf(timestamp.timestamp())
    metadata[f"{name}_s"] = int(s)
    metadata[f"{name}_ns"] = int(f * NS_PER_SEC)


class InstrumentedTimeProfiler(holoscan.core.Operator):
    def __init__(
        self,
        *args,
        recorder_queue=None,
        operator_name="operator",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._recorder_queue = recorder_queue
        self._operator_name = operator_name

    def setup(self, spec):
        logging.info("setup")
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        # What time is it now?
        operator_timestamp = datetime.datetime.utcnow()

        in_message = op_input.receive("input")
        cp_frame = cp.asarray(in_message.get(""))
        #
        save_timestamp(
            self.metadata, self._operator_name + "_timestamp", operator_timestamp
        )
        op_output.emit({"": cp_frame}, "output")


class MonitorOperator(holoscan.core.Operator):
    def __init__(
        self,
        *args,
        recorder_queue=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._recorder_queue = recorder_queue

    def setup(self, spec):
        logging.info("setup")
        spec.input("input")

    def compute(self, op_input, op_output, context):
        # What time is it now?
        complete_timestamp = datetime.datetime.utcnow()

        _ = op_input.receive("input")
        #
        save_timestamp(self.metadata, "complete_timestamp", complete_timestamp)
        record_times(self._recorder_queue, self.metadata)
