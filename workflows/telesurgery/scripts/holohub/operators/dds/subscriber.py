import rti.connextdds as dds
from holoscan.core import Operator
from holoscan.core._core import OperatorSpec


class DDSSubscriberOp(Operator):
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
        self.dds_reader = None
        self.selector = None

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def start(self):
        dp = dds.DomainParticipant(domain_id=self.dds_domain_id)
        self.dds_reader = dds.DataReader(dds.Topic(dp, self.dds_topic, self.dds_topic_class))
        # self.selector = self.dds_reader.select().max_samples(1)
        print(f"Reading data from DDS: {self.dds_topic}:{self.dds_domain_id} => {self.dds_topic_class.__name__}")

    def compute(self, op_input, op_output, context):
        for data in self.dds_reader.take_data():
            op_output.emit(data, "output")
