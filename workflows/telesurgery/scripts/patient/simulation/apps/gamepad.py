import argparse

from holohub.operators.dds.subscriber import DDSSubscriberOp
from holohub.operators.mira.gamepad.controller import GamepadControllerOp
from holoscan.core import Application
from patient.simulation.annotators.mira import ApiControllerOp
from schemas.gamepad_event import GamepadEvent


class App(Application):
    """Application to capture room camera and process its output."""

    def __init__(
        self,
        dds_domain_id,
        dds_topic,
        callback,
    ):
        self.dds_domain_id = dds_domain_id
        self.dds_topic = dds_topic
        self.callback = callback

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

        server = ApiControllerOp(
            self,
            name="api_controller",
            callback=self.callback,
        )

        self.add_flow(dds, gamepad, {("output", "input")})
        self.add_flow(gamepad, server, {("output", "input")})


def main():
    """Parse command-line arguments and run the application."""
    parser = argparse.ArgumentParser(description="Run the hid application")
    parser.add_argument("--domain_id", type=int, default=779, help="dds domain id")
    parser.add_argument("--topic", type=str, default="telesurgery/hid/gamepad", help="dds topic name")

    args = parser.parse_args()
    app = App(
        dds_domain_id=args.domain_id,
        dds_topic=args.topic,
        callback=None
    )
    app.run()


if __name__ == "__main__":
    main()
