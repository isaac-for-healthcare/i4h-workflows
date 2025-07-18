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
from typing import Dict, Any, List

from simulation.communication.interface import CommunicationServer, CommunicationData


class TCPServer(CommunicationServer):
    """TCP server implementation for hardware driver."""
    
    def __init__(self, host: str = "localhost", port: int = 8888, **kwargs):
        """
        Initialize TCP server.
        
        Args:
            host: Server host address
            port: Server port number
        """
        super().__init__(**kwargs)
        self.host = host
        self.port = port
        self.server_socket: socket.socket = None
        self.client_sockets: List[socket.socket] = []
        self.client_threads: List[threading.Thread] = []
        
    def start_server(self, **kwargs) -> bool:
        """
        Start the TCP server.
        
        Returns:
            bool: True if server started successfully
        """
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            
            # Start accept thread
            accept_thread = threading.Thread(target=self._accept_clients)
            accept_thread.daemon = True
            accept_thread.start()
            
            return True
            
        except Exception as e:
            if self.error_callback:
                self.error_callback(f"Failed to start TCP server: {e}")
            return False
    
    def stop_server(self) -> None:
        """Stop the TCP server."""
        self.running = False
        
        # Close all client connections
        for client_socket in self.client_sockets:
            try:
                client_socket.close()
            except:
                pass
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        # Wait for threads to finish
        for thread in self.client_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        self.client_sockets.clear()
        self.client_threads.clear()
    
    def broadcast_data(self, data: CommunicationData) -> bool:
        """
        Broadcast data to all connected clients.
        
        Args:
            data: Data to broadcast
            
        Returns:
            bool: True if broadcast successful
        """
        if not self.running:
            return False
        
        message = json.dumps(data.to_dict()) + '\n'
        
        # Send to all connected clients
        disconnected_clients = []
        
        for client_socket in self.client_sockets:
            try:
                client_socket.send(message.encode('utf-8'))
            except:
                # Mark client for removal
                disconnected_clients.append(client_socket)
        
        # Remove disconnected clients
        for client_socket in disconnected_clients:
            self._remove_client(client_socket)
        
        return True
    
    def get_protocol_name(self) -> str:
        """Get the name of the communication protocol."""
        return "TCP"
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information."""
        return {
            "protocol": "TCP",
            "host": self.host,
            "port": self.port,
            "running": self.running,
            "connected_clients": len(self.client_sockets)
        }
    
    def _accept_clients(self) -> None:
        """Accept new client connections."""
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                
                # Add client to list
                self.client_sockets.append(client_socket)
                
                # Start client handler thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, addr)
                )
                client_thread.daemon = True
                client_thread.start()
                self.client_threads.append(client_thread)
                
            except Exception as e:
                if self.running and self.error_callback:
                    self.error_callback(f"Error accepting client: {e}")
    
    def _handle_client(self, client_socket: socket.socket, addr) -> None:
        """
        Handle individual client connection.
        
        Args:
            client_socket: Client socket
            addr: Client address
        """
        try:
            # Keep connection alive while server is running
            while self.running:
                try:
                    # Set a timeout so we can check if server is still running
                    client_socket.settimeout(1.0)
                    
                    # Try to receive data (this will timeout if no data)
                    # We don't expect clients to send data in this use case,
                    # but we need to keep the connection alive
                    try:
                        data = client_socket.recv(1024)
                        if not data:
                            # Client disconnected gracefully
                            break
                        
                        # If client sends data, we could process it here
                        # For now, we just ignore it since we're broadcasting
                        
                    except socket.timeout:
                        # Timeout is expected, continue to check if server is running
                        continue
                    except socket.error:
                        # Client disconnected unexpectedly
                        break
                        
                except Exception as e:
                    break
                    
        except Exception as e:
            if self.error_callback:
                self.error_callback(f"Error handling client {addr}: {e}")
        finally:
            self._remove_client(client_socket)
    
    def _remove_client(self, client_socket: socket.socket) -> None:
        """Remove a client from the server."""
        try:
            client_socket.close()
        except:
            pass
        
        if client_socket in self.client_sockets:
            self.client_sockets.remove(client_socket)
