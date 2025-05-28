import argparse

from holohub.operators.mira.api_client import ApiClientOp
from holohub.operators.mira.haply.controller import HaplyControllerOp
from holoscan.core import Application


class App(Application):
    """Application to run the HID operator and process its input."""

    def __init__(self, uri, api_host, api_port):
        self.uri = uri
        self.api_host = api_host
        self.api_port = api_port

        super().__init__()

    def compose(self):
        haply = HaplyControllerOp(
            self,
            name="haply_controller",
            uri=self.uri,
        )
        client = ApiClientOp(
            self,
            name="api_client",
            host=self.api_host,
            port=self.api_port,
        )

        self.add_flow(haply, client, {("output", "input")})


def main():
    """Parse command-line arguments and run the application."""
    parser = argparse.ArgumentParser(description="Run the hid application")
    parser.add_argument("--uri", type=str, default="ws://localhost:10001", help="haply inverse local uri")
    parser.add_argument("--api_host", type=str, default="10.137.145.1", help="api server host")
    parser.add_argument("--api_port", type=int, default=8081, help="api server port")

    args = parser.parse_args()

    app = App(
        uri=args.uri,
        api_host=args.api_host,
        api_port=args.api_port,
    )
    app.run()


if __name__ == "__main__":
    main()
