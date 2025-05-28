import argparse

from holohub.operators.mira.api_server import ApiServerOp
from holoscan.core import Application
from patient.simulation.mira.handler import ApiHandlerOp


class App(Application):
    """Application to capture room camera and process its output."""

    def __init__(self, api_host, api_port, callback):
        self.api_host = api_host
        self.api_port = api_port
        self.callback = callback

        super().__init__()

    def compose(self):
        server = ApiServerOp(
            self,
            host=self.api_host,
            port=self.api_port,
        )

        handler = ApiHandlerOp(
            self,
            name="api_handler",
            callback=self.callback,
        )

        self.add_flow(server, handler, {("output", "input")})


def main():
    """Parse command-line arguments and run the application."""
    parser = argparse.ArgumentParser(description="Run MIRA API server (Simulation)")
    parser.add_argument("--api_host", type=str, default="0.0.0.0", help="api server host")
    parser.add_argument("--api_port", type=int, default=8081, help="api server port")

    args = parser.parse_args()
    app = App(api_host=args.api_host, api_port=args.api_port, callback=None)
    app.run()


if __name__ == "__main__":
    main()
