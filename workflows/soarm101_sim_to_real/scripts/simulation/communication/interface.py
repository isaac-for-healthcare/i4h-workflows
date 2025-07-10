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


import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass


@dataclass
class CommunicationData:
    """Data structure for communication between components."""
    
    leader_arm: Dict[str, float]
    follower_arm: Optional[Dict[str, float]] = None
    hardware_status: Optional[Dict[str, Any]] = None
    timestamp: float = 0.0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommunicationData':
        """Create CommunicationData from dictionary."""
        return cls(
            leader_arm=data.get('leader_arm', {}),
            follower_arm=data.get('follower_arm'),
            hardware_status=data.get('status', data.get('hardware_status')),
            timestamp=data.get('timestamp', time.time())
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert CommunicationData to dictionary."""
        return {
            'leader_arm': self.leader_arm,
            'follower_arm': self.follower_arm,
            'status': self.hardware_status,
            'timestamp': self.timestamp
        }


class CommunicationInterface(ABC):
    """Abstract base class for communication protocols."""
    
    def __init__(self, **kwargs):
        """Initialize communication interface."""
        self.connected = False
        self.data_callback: Optional[Callable[[CommunicationData], None]] = None
        self.error_callback: Optional[Callable[[str], None]] = None
    
    @abstractmethod
    def connect(self, **kwargs) -> bool:
        """Connect to communication endpoint."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from communication endpoint."""
        pass
    
    @abstractmethod
    def receive_data(self, timeout: float = 1.0) -> Optional[CommunicationData]:
        """Receive data from communication channel."""
        pass
    
    @abstractmethod
    def send_data(self, data: CommunicationData) -> bool:
        """Send data through communication channel."""
        pass
    
    @abstractmethod
    def get_protocol_name(self) -> str:
        """Get the name of the communication protocol."""
        pass
    
    @abstractmethod
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information."""
        pass
    
    def is_connected(self) -> bool:
        """Check if communication is connected."""
        return self.connected
    
    def set_data_callback(self, callback: Callable[[CommunicationData], None]) -> None:
        """Set callback for received data."""
        self.data_callback = callback
    
    def set_error_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for errors."""
        self.error_callback = callback


class CommunicationServer(ABC):
    """Abstract base class for communication servers."""
    
    def __init__(self, **kwargs):
        """Initialize communication server."""
        self.running = False
        self.clients = []
        self.data_callback: Optional[Callable[[CommunicationData], None]] = None
        self.error_callback: Optional[Callable[[str], None]] = None
    
    @abstractmethod
    def start_server(self, **kwargs) -> bool:
        """Start the communication server."""
        pass
    
    @abstractmethod
    def stop_server(self) -> None:
        """Stop the communication server."""
        pass
    
    @abstractmethod
    def broadcast_data(self, data: CommunicationData) -> bool:
        """Broadcast data to all connected clients."""
        pass
    
    @abstractmethod
    def get_protocol_name(self) -> str:
        """Get the name of the communication protocol."""
        pass
    
    @abstractmethod
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information."""
        pass
    
    def is_running(self) -> bool:
        """Check if server is running."""
        return self.running
    
    def set_data_callback(self, callback: Callable[[CommunicationData], None]) -> None:
        """Set callback for received data."""
        self.data_callback = callback
    
    def set_error_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for errors."""
        self.error_callback = callback
