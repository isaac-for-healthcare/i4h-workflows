import argparse

from holohub.operators.dds.publisher import DDSPublisherOp
from holohub.operators.hid.gamepad import GamepadOp
from holohub.operators.sink import NoOp
from holoscan.core import Application
from schemas.gamepad_event import GamepadEvent


class App(Application):
    """Application to run the HID operator and process its input."""

    def __init__(self, device_idx, dds_domain_id, dds_topic):
        self.device_idx = device_idx
        self.dds_domain_id = dds_domain_id
        self.dds_topic = dds_topic

        super().__init__()

    def compose(self):
        gamepad = GamepadOp(
            self,
            name="gamepad",
            device_idx=self.device_idx,
        )
        dds = DDSPublisherOp(
            self,
            name="dds_publisher",
            dds_domain_id=self.dds_domain_id,
            dds_topic=self.dds_topic,
            dds_topic_class=GamepadEvent,
        )
        sink = NoOp(self)

        self.add_flow(gamepad, dds, {("output", "input")})
        self.add_flow(dds, sink, {("output", "input")})


def main():
    """Parse command-line arguments and run the application."""
    parser = argparse.ArgumentParser(description="Run the hid application")
    parser.add_argument("--device_idx", type=int, default=0, help="device index case of multiple joysticks")
    parser.add_argument("--domain_id", type=int, default=779, help="dds domain id")
    parser.add_argument("--topic", type=str, default="telesurgery/hid/gamepad", help="dds topic name")

    args = parser.parse_args()

    app = App(
        device_idx=args.device_idx,
        dds_domain_id=args.domain_id,
        dds_topic=args.topic,
    )
    app.run()


if __name__ == "__main__":
    main()
