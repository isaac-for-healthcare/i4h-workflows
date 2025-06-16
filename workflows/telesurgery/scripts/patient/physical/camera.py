# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import ctypes
import json
import os

import hololink
from cuda import cuda
from holohub.operators.camera.cv2 import CV2ToCameraStreamOp
from holohub.operators.camera.hsb import HSBToCameraStreamOp
from holohub.operators.camera.realsense import RealsenseToCameraStreamOp
from holohub.operators.dds.publisher import DDSPublisherOp
from holohub.operators.nvidia_video_codec.utils.camera_stream_merge import CameraStreamMergeOp
from holohub.operators.nvidia_video_codec.utils.camera_stream_split import CameraStreamSplitOp
from holohub.operators.nvjpeg.encoder import NVJpegEncoderOp
from holohub.operators.stats import CameraStreamStats
from holohub.operators.to_viz import CameraStreamToViz
from holoscan.conditions import BooleanCondition
from holoscan.core import Application, MetadataPolicy
from holoscan.operators import BayerDemosaicOp, HolovizOp
from holoscan.resources import BlockMemoryPool, MemoryStorageType
from schemas.camera_stream import CameraStream


class App(Application):
    """Application to capture room camera and process its output."""

    def __init__(
        self,
        camera: str,
        camera_name: str,
        width: int,
        height: int,
        device_idx: int,
        framerate: int,
        stream_type,
        stream_format,
        dds_domain_id,
        dds_topic,
        encoder,
        encoder_params,
        hsb_cuda_context=None,
        hsb_cuda_device_ordinal=None,
        hsb_hololink_channel=None,
        hsb_ibv_name=None,
        hsb_ibv_port=None,
        hsb_camera=None,
        show_viz=False,
        srgb=True,
    ):
        self.camera = camera
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.device_idx = device_idx
        self.framerate = framerate
        self.stream_type = stream_type
        self.stream_format = stream_format
        self.dds_domain_id = dds_domain_id
        self.dds_topic = dds_topic
        self.encoder = encoder
        self.encoder_params = encoder_params

        self.hsb_cuda_context = hsb_cuda_context
        self.hsb_cuda_device_ordinal = hsb_cuda_device_ordinal
        self.hsb_hololink_channel = hsb_hololink_channel
        self.hsb_ibv_name = hsb_ibv_name
        self.hsb_ibv_port = hsb_ibv_port
        self.hsb_camera = hsb_camera
        self.show_viz = show_viz
        self.srgb = srgb

        super().__init__()

    def compose(self):
        if self.camera == "hsb":
            condition = BooleanCondition(self, name="ok", enable_tick=True)
            csi_to_bayer_pool = BlockMemoryPool(
                self,
                name="pool",
                # storage_type of 1 is device memory
                storage_type=1,
                block_size=self.hsb_camera._width * ctypes.sizeof(ctypes.c_uint16) * self.hsb_camera._height,
                num_blocks=2,
            )
            csi_to_bayer_operator = hololink.operators.CsiToBayerOp(
                self,
                name="csi_to_bayer",
                allocator=csi_to_bayer_pool,
                cuda_device_ordinal=self.hsb_cuda_device_ordinal,
            )
            self.hsb_camera.configure_converter(csi_to_bayer_operator)

            frame_size = csi_to_bayer_operator.get_csi_length()
            frame_context = self.hsb_cuda_context
            receiver_operator = hololink.operators.RoceReceiverOp(
                self,
                condition,
                name="receiver",
                frame_size=frame_size,
                frame_context=frame_context,
                ibv_name=self.hsb_ibv_name,
                ibv_port=self.hsb_ibv_port,
                hololink_channel=self.hsb_hololink_channel,
                device=self.hsb_camera,
            )

            bayer_format = self.hsb_camera.bayer_format()
            pixel_format = self.hsb_camera.pixel_format()
            image_processor_operator = hololink.operators.ImageProcessorOp(
                self,
                name="image_processor",
                # Optical black value for imx274 is 50
                optical_black=50,
                bayer_format=bayer_format.value,
                pixel_format=pixel_format.value,
            )

            rgba_components_per_pixel = 4
            bayer_pool = BlockMemoryPool(
                self,
                name="pool",
                # storage_type of 1 is device memory
                storage_type=1,
                block_size=self.hsb_camera._width
                * rgba_components_per_pixel
                * ctypes.sizeof(ctypes.c_uint16)
                * self.hsb_camera._height,
                num_blocks=2,
            )
            demosaic = BayerDemosaicOp(
                self,
                name="demosaic",
                pool=bayer_pool,
                generate_alpha=True,
                alpha_value=65535,
                bayer_grid_pos=bayer_format.value,
                interpolation_mode=0,
            )
            image_shift = hololink.operators.ImageShiftToUint8Operator(self, name="image_shift", shift=8)
            source = HSBToCameraStreamOp(self, name="hsb", width=self.width, height=self.height)

            self.add_flow(receiver_operator, csi_to_bayer_operator, {("output", "input")})
            self.add_flow(csi_to_bayer_operator, image_processor_operator, {("output", "input")})
            self.add_flow(image_processor_operator, demosaic, {("output", "receiver")})
            self.add_flow(demosaic, image_shift, {("transmitter", "input")})
            self.add_flow(image_shift, source, {("output", "input")})

        elif self.camera == "realsense":
            source = RealsenseToCameraStreamOp(
                self,
                name=self.camera_name,
                width=self.width,
                height=self.height,
                device_idx=self.device_idx,
                framerate=self.framerate,
                stream_type=self.stream_type,
                stream_format=self.stream_format,
            )
        else:
            source = CV2ToCameraStreamOp(
                self,
                name=self.camera_name,
                width=self.width,
                height=self.height,
                device_idx=self.device_idx,
                framerate=self.framerate,
            )

        if self.encoder == "nvc":
            try:
                from holohub.operators.nvidia_video_codec.nv_video_encoder import NvVideoEncoderOp

                encoder_op = NvVideoEncoderOp(
                    self,
                    name="nvc_encoder",
                    cuda_device_ordinal=0,
                    width=self.width,
                    height=self.height,
                    codec=self.encoder_params.get("codec", "H264"),
                    preset=self.encoder_params.get("preset", "P3"),
                    bitrate=self.encoder_params.get("bitrate", 10000000),
                    frame_rate=self.encoder_params.get("frame_rate", self.framerate),
                    rate_control_mode=self.encoder_params.get("rate_control_mode", 1),
                    multi_pass_encoding=self.encoder_params.get("multi_pass_encoding", 0),
                    allocator=BlockMemoryPool(
                        self,
                        name="pool",
                        storage_type=MemoryStorageType.HOST,
                        block_size=self.width * self.height * 4,
                        num_blocks=2,
                    ),
                )
            except Exception as e:
                print(f"Error creating NVC encoder: {e}")
                raise e
            split_op = CameraStreamSplitOp(self, name="split_op")
            merge_op = CameraStreamMergeOp(self, name="merge_op")
            merge_op.metadata_policy = MetadataPolicy.UPDATE
        else:
            encoder_op = NVJpegEncoderOp(
                self,
                name="nvjpeg_encoder",
                skip=self.encoder != "nvjpeg",
                quality=self.encoder_params.get("quality", 90),
            )

        dds = DDSPublisherOp(
            self,
            name="dds_publisher",
            dds_domain_id=self.dds_domain_id,
            dds_topic=self.dds_topic,
            dds_topic_class=CameraStream,
        )
        stats = CameraStreamStats(self, name="stats", interval_ms=1000)

        if self.show_viz:
            stream_to_viz = CameraStreamToViz(self)
            visualizer = HolovizOp(self, name="holoviz", framebuffer_srgb=True)
            self.add_flow(source, stream_to_viz, {("output", "input")})
            self.add_flow(stream_to_viz, visualizer, {("output", "receivers")})

        if self.encoder == "nvc":
            print("Using NVC encoder with split and merge")
            self.add_flow(source, split_op, {("output", "input")})
            self.add_flow(split_op, merge_op, {("output", "input")})
            self.add_flow(split_op, encoder_op, {("image", "input")})
            self.add_flow(encoder_op, merge_op, {("output", "image")})
            self.add_flow(merge_op, dds, {("output", "input")})
            self.add_flow(merge_op, stats, {("output", "input")})
        else:
            self.add_flow(source, encoder_op, {("output", "input")})
            self.add_flow(encoder_op, dds, {("output", "input")})
            self.add_flow(encoder_op, stats, {("output", "input")})


def main():
    """Parse command-line arguments and run the application."""
    parser = argparse.ArgumentParser(description="Run the camera application")
    parser.add_argument(
        "--camera", type=str, default="imx274", choices=["realsense", "cv2", "imx274"], help="camera type"
    )
    parser.add_argument("--hololink", type=str, default="192.168.0.2", help="IP address of Hololink board")
    parser.add_argument("--name", type=str, default="robot", help="camera name")
    parser.add_argument("--width", type=int, default=1920, help="width")
    parser.add_argument("--height", type=int, default=1080, help="height")
    parser.add_argument("--device_idx", type=int, default=0, help="device index")
    parser.add_argument("--framerate", type=int, default=60, help="frame rate")
    parser.add_argument("--stream_type", type=str, default="color", choices=["color", "depth"])
    parser.add_argument("--stream_format", type=str, default="")
    parser.add_argument("--encoder", type=str, choices=["nvjpeg", "nvc", "none"], default="nvc", help="encoder type")
    parser.add_argument("--encoder_params", type=str, default=None, help="encoder params")
    parser.add_argument("--domain_id", type=int, default=9, help="dds domain id")
    parser.add_argument("--topic", type=str, default="", help="dds topic name")
    parser.add_argument("--show_viz", type=bool, default=False, help="show viz")
    parser.add_argument("--srgb", type=bool, default=True, help="framebuffer srgb for viz")

    infiniband_devices = hololink.infiniband_devices()
    parser.add_argument("--ibv-name", default=infiniband_devices[0], help="IBV device to use")
    parser.add_argument("--ibv-port", type=int, default=1, help="Port number of IBV device")

    args = parser.parse_args()

    cu_device = None
    cu_device_ordinal = None
    cu_context = None
    hololink_channel = None
    hsb_hololink = None
    hsb_camera = None
    if args.camera == "imx274":
        assert args.framerate == 60

        camera_mode = -1
        if args.width == 3840 and args.height == 2160:
            camera_mode = 0
        elif args.width == 1920 and args.height == 1080:
            camera_mode = 1
        if camera_mode < 0:
            raise f"(only 3840x2160 or 1920x1080 are supported for {args.camera}"

        args.camera = "hsb"

        (cu_result,) = cuda.cuInit(0)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS

        cu_device_ordinal = 0
        cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS

        cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS

        # Get a handle to the data source
        channel_metadata = hololink.Enumerator.find_channel(channel_ip=args.hololink)
        print(f"{channel_metadata=}")
        hololink_channel = hololink.DataChannel(channel_metadata)

        # Get a handle to the camera
        hsb_camera = hololink.sensors.imx274.dual_imx274.Imx274Cam(hololink_channel, expander_configuration=0)
        hsb_camera_mode = hololink.sensors.imx274.imx274_mode.Imx274_Mode(camera_mode)
        hsb_camera.set_mode(hsb_camera_mode)

    if args.encoder == "nvjpeg" and args.encoder_params is None:
        encoder_params = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nvjpeg_encoder_params.json")
    elif args.encoder == "nvc" and args.encoder_params is None:
        encoder_params = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nvc_encoder_params.json")
    elif args.encoder_params and os.path.isfile(args.encoder_params):
        encoder_params = args.encoder_params
    else:
        encoder_params = json.loads(args.encoder_params) if args.encoder_params else {}

    print(f"Encoder params: {encoder_params}")
    if isinstance(encoder_params, str) and os.path.isfile(encoder_params):
        with open(encoder_params) as f:
            encoder_params = json.load(f)

    app = App(
        camera=args.camera,
        camera_name=args.name,
        width=args.width,
        height=args.height,
        device_idx=args.device_idx,
        framerate=args.framerate,
        stream_type=args.stream_type,
        stream_format=args.stream_format,
        encoder=args.encoder,
        encoder_params=encoder_params,
        dds_domain_id=args.domain_id,
        dds_topic=args.topic if args.topic else f"telesurgery/{args.name}_camera/rgb",
        hsb_cuda_context=cu_context,
        hsb_cuda_device_ordinal=cu_device_ordinal,
        hsb_hololink_channel=hololink_channel,
        hsb_ibv_name=args.ibv_name,
        hsb_ibv_port=args.ibv_port,
        hsb_camera=hsb_camera,
        show_viz=args.show_viz,
        srgb=args.srgb,
    )

    if hololink_channel is not None:
        hsb_hololink = hololink_channel.hololink()
        hsb_hololink.start()

    app.run()

    if hsb_hololink is not None:
        hsb_hololink.stop()
    if cu_device is not None:
        (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS


if __name__ == "__main__":
    main()
