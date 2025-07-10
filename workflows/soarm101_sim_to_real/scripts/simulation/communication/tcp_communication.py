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


import socket
import json
import threading
from queue import Queue, Empty
from typing import Dict, Any, Optional

from simulation.communication.interface import CommunicationInterface, CommunicationData

class TCPCommunication(CommunicationInterface):
    """TCP client communication implementation."""
    
    def __init__(self, host: str = "localhost", port: int = 8888, **kwargs):
        """
        Initialize TCP communication.
        
        Args:
            host: TCP server hostname or IP
            port: TCP server port
        """
        super().__init__(**kwargs)
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.receive_thread: Optional[threading.Thread] = None
        self.data_queue = Queue()
        self.running = False
        
    def connect(self, **kwargs) -> bool:
        """
        Connect to TCP server.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.host, self.port))
            
            self.connected = True
            self.running = True
            
            # Start receiving thread
            self.receive_thread = threading.Thread(target=self._receive_loop)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            
            return True
            
        except Exception as e:
            self.connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from TCP server."""
        self.running = False
        self.connected = False
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
            
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=1.0)
    
    def _receive_loop(self) -> None:
        """Background thread for receiving data."""
        buffer = ""
        
        while self.running and self.socket:
            try:
                data = self.socket.recv(1024).decode('utf-8')
                if not data:
                    break
                    
                buffer += data
                
                # Process complete JSON messages
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            json_data = json.loads(line.strip())
                            comm_data = CommunicationData.from_dict(json_data)
                            self.data_queue.put(comm_data)
                            
                            # Call data callback if set
                            if self.data_callback:
                                self.data_callback(comm_data)
                                
                        except json.JSONDecodeError as e:
                            if self.error_callback:
                                self.error_callback(f"Invalid JSON: {e}")
                            
            except socket.timeout:
                continue  # Keep trying
            except Exception as e:
                if self.error_callback:
                    self.error_callback(f"TCP receive error: {e}")
                break
        
        self.connected = False
    
    def receive_data(self, timeout: float = 1.0) -> Optional[CommunicationData]:
        """
        Receive data from TCP server.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            CommunicationData or None if no data available
        """
        try:
            return self.data_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def send_data(self, data: CommunicationData) -> bool:
        """
        Send data to TCP server.
        
        Args:
            data: Data to send
            
        Returns:
            bool: True if send successful, False otherwise
        """
        if not self.connected or not self.socket:
            return False
            
        try:
            json_str = json.dumps(data.to_dict()) + '\n'
            self.socket.send(json_str.encode('utf-8'))
            return True
        except Exception as e:
            if self.error_callback:
                self.error_callback(f"TCP send error: {e}")
            return False
    
    def get_protocol_name(self) -> str:
        """Get the name of the communication protocol."""
        return "TCP"
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information."""
        return {
            "protocol": "TCP",
            "host": self.host,
            "port": self.port,
            "connected": self.connected
        }
    
    def get_latest_data(self) -> Optional[CommunicationData]:
        """
        Get the latest data from queue, discarding older data.
        
        Returns:
            CommunicationData or None if no data available
        """
        latest_data = None
        
        # Get all available data, keep only the latest
        while True:
            try:
                latest_data = self.data_queue.get_nowait()
            except Empty:
                break
                
        return latest_data
    
    def clear_queue(self) -> None:
        """Clear the data queue."""
        while True:
            try:
                self.data_queue.get_nowait()
            except Empty:
                break
