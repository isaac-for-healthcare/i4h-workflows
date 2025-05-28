import argparse

from holohub.operators.dds.subscriber import DDSSubscriberOp
from holohub.operators.mira.api_client import ApiClientOp
from holohub.operators.mira.gamepad.controller import GamepadControllerOp
from holoscan.core import Application
from schemas.gamepad_event import GamepadEvent


class App(Application):
    """Application to capture room camera and process its output."""

    def __init__(
        self,
        dds_domain_id,
        dds_topic,
        api_host,
        api_port,
    ):
        self.dds_domain_id = dds_domain_id
        self.dds_topic = dds_topic
        self.api_host = api_host
        self.api_port = api_port

        super().__init__()

    def compose(self):
        dds = DDSSubscriberOp(
            self,
            name="dds_subscriber",
            dds_domain_id=self.dds_domain_id,
            dds_topic=self.dds_topic,
            dds_topic_class=GamepadEvent,
        )

        gamepad = GamepadControllerOp(
            self,
            name="gamepad_controller",
        )

        client = ApiClientOp(
            self,
            name="api_client",
            host=self.api_host,
            port=self.api_port,
        )

        self.add_flow(dds, gamepad, {("output", "input")})
        self.add_flow(gamepad, client, {("output", "input")})


def main():
    """Parse command-line arguments and run the application."""
    parser = argparse.ArgumentParser(description="Run the hid application")
    parser.add_argument("--domain_id", type=int, default=779, help="dds domain id")
    parser.add_argument("--topic", type=str, default="telesurgery/hid/gamepad", help="dds topic name")
    parser.add_argument("--api_host", type=str, default="10.137.145.1", help="api server host")
    parser.add_argument("--api_port", type=int, default=8081, help="api server port")

    args = parser.parse_args()
    app = App(
        dds_domain_id=args.domain_id,
        dds_topic=args.topic,
        api_host=args.api_host,
        api_port=args.api_port,
    )
    app.run()


if __name__ == "__main__":
    main()
