from abc import ABC
from typing import Any, Dict, Tuple

import rti.connextdds as dds  # noqa: F401


class DDSEntity(ABC):
    """
    Base class for DDS entities (Publishers and Subscribers).

    Args:
        topic: The DDS topic name.
        cls: The class type of the data.
        period: Time period between successive operations in seconds.
        domain_id: The DDS domain ID.
        qos_provider_path: Path to XML file containing QoS profiles.
        transport_profile: Transport QoS profile name (format: "Library::Profile").
        entity_profile: Entity-specific QoS profile name (format: "Library::Profile").
    """
    # Cache for QoS profiles - shared across all instances
    _QOS_CACHE: Dict[str, Dict[str, Tuple[dds.DomainParticipantQos, dds.TopicQos, Any]]] = {}

    def __init__(
        self,
        topic: str,
        cls: Any,
        period: float,
        domain_id: int,
        qos_provider_path: str,
        transport_profile: str,
        entity_profile: str
    ):
        self.topic = topic
        self.cls = cls
        self.period = period
        self.domain_id = domain_id
        self.qos_provider_path = qos_provider_path
        self.transport_profile = transport_profile
        self.entity_profile = entity_profile

    def _get_cached_qos(self, profile_name: str) -> Tuple[dds.DomainParticipantQos, dds.TopicQos, Any]:
        """
        Get QoS settings from cache or load from XML file.

        Args:
            profile_name: QoS profile name (format: "Library::Profile").

        Returns:
            Tuple containing participant, topic, and entity-specific QoS settings.
        """
        # Check if provider is cached
        if self.qos_provider_path not in DDSEntity._QOS_CACHE:
            DDSEntity._QOS_CACHE[self.qos_provider_path] = {}
        
        provider_cache = DDSEntity._QOS_CACHE[self.qos_provider_path]
        
        # Check if profile is cached
        if profile_name not in provider_cache:
            provider = dds.QosProvider(self.qos_provider_path)
            provider_cache[profile_name] = (
                provider.participant_qos_from_profile(profile_name),
                provider.topic_qos_from_profile(profile_name),
                self._get_entity_qos(provider, profile_name)
            )
        
        return provider_cache[profile_name]

    def _get_entity_qos(self, provider: dds.QosProvider, profile_name: str) -> Any:
        """
        Get entity-specific QoS settings from provider.
        To be implemented by derived classes.

        Args:
            provider: The DDS QoS provider.
            profile_name: QoS profile name.

        Returns:
            Entity-specific QoS settings.
        """
        raise NotImplementedError("Derived classes must implement _get_entity_qos")

    def _create_participant(self) -> dds.DomainParticipant:
        """Create a DDS domain participant with cached QoS settings."""
        participant_qos, _, _ = self._get_cached_qos(self.transport_profile)
        return dds.DomainParticipant(
            domain_id=self.domain_id,
            qos=participant_qos
        )

    def _create_topic(self, participant: dds.DomainParticipant) -> dds.Topic:
        """Create a DDS topic with cached QoS settings."""
        _, topic_qos, _ = self._get_cached_qos(self.entity_profile)
        return dds.Topic(
            participant,
            self.topic,
            self.cls,
            qos=topic_qos
        ) 