import rti.connextdds as dds
from holoscan.core import Operator
from holoscan.core._core import OperatorSpec


class DDSPublisherOp(Operator):
    """
    Operator to write data to DDS.
    """

    def __init__(self, fragment, dds_domain_id, dds_topic, dds_topic_class, *args, **kwargs):
        """
        Initialize the DDSPublisherOp operator.

        Parameters:
        - dds_domain_id (int): DDS domain ID.
        - dds_topic (str): DDS topic for frames.
        - dds_topic_class (class): DDS class that represents the topic.
        """
        self.dds_domain_id = dds_domain_id
        self.dds_topic = dds_topic
        self.dds_topic_class = dds_topic_class

        super().__init__(fragment, *args, **kwargs)
        self.dds_writer = None

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output")

    def start(self):
        dp = dds.DomainParticipant(domain_id=self.dds_domain_id)
        self.dds_writer = dds.DataWriter(dp.implicit_publisher, dds.Topic(dp, self.dds_topic, self.dds_topic_class))
        print(f"Writing data to DDS: {self.dds_topic}:{self.dds_domain_id} => {self.dds_topic_class.__name__}")

    def compute(self, op_input, op_output, context):
        stream = op_input.receive("input")
        assert isinstance(stream, self.dds_topic_class)

        self.dds_writer.write(stream)
        op_output.emit(stream, "output")
