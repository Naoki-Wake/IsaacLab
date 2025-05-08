# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import numpy as np

from pydantic import BaseModel
from typing import Any, Dict

import zmq
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict
import torch
import time
from abc import ABC, abstractmethod


class ModalityConfig(BaseModel):
    """Configuration for a modality."""
    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""

class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method to get the action for a given state.

        Args:
            observations: The observations from the environment.

        Returns:
            The action to take in the environment in dictionary format.
        """
        raise NotImplementedError

    @abstractmethod
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Return the modality config of the policy.
        """
        raise NotImplementedError

class TorchSerializer:
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        buffer = BytesIO(data)
        import sys, types
        gr00t = types.ModuleType("gr00t")
        sys.modules["gr00t"] = gr00t

        # Fake "gr00t.data" subpackage
        gr00t_data = types.ModuleType("gr00t.data")
        sys.modules["gr00t.data"] = gr00t_data
        gr00t_data_dataset = types.ModuleType("gr00t.data.dataset")
        sys.modules["gr00t.data.dataset"] = gr00t_data_dataset

        gr00t_data_dataset.ModalityConfig = ModalityConfig
        buffer = BytesIO(data)
        obj = torch.load(buffer, weights_only=False)
        return obj


class BaseInferenceClient:
    def __init__(self, host: str = "10.137.70.15", port: int = 5555, timeout_ms: int = 15000):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        """
        Kill the server.
        """
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> dict:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data

        self.socket.send(TorchSerializer.to_bytes(request))
        message = self.socket.recv()
        if message == b"ERROR":
            raise RuntimeError("Server error")
        return TorchSerializer.from_bytes(message)

    def __del__(self):
        """Cleanup resources on destruction"""
        self.socket.close()
        self.context.term()


class RobotInferenceClient(BaseInferenceClient, BasePolicy):
    """
    Client for communicating with the RealRobotServer
    """

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        return self.call_endpoint("get_action", observations)

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        return self.call_endpoint("get_modality_config", requires_input=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, help="Port number for the server.", default=5555)
    parser.add_argument(
        "--host", type=str, help="Host address for the server.", default="localhost"
    )
    args = parser.parse_args()

    # In this mode, we will send a random observation to the server and get an action back
    # This is useful for testing the server and client connection
    # Create a policy wrapper
    policy_client = RobotInferenceClient(host=args.host, port=args.port)

    print("Available modality config available:")
    modality_configs = policy_client.get_modality_config()
    print(modality_configs.keys())

    # Making prediction...
    # - obs: video.ego_view: (1, 256, 256, 3)
    # - obs: state.left_arm: (1, 7)
    # - obs: state.right_arm: (1, 7)
    # - obs: state.left_hand: (1, 6)
    # - obs: state.right_hand: (1, 6)
    # - obs: state.waist: (1, 3)

    # - action: action.left_arm: (16, 7)
    # - action: action.right_arm: (16, 7)
    # - action: action.left_hand: (16, 6)
    # - action: action.right_hand: (16, 6)
    # - action: action.waist: (16, 3)
    obs = {
        "video.ego_view": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
        "state.left_arm": np.random.rand(1, 7),
        "state.right_arm": np.random.rand(1, 7),
        "state.left_hand": np.random.rand(1, 6),
        "state.right_hand": np.random.rand(1, 6),
        "state.waist": np.random.rand(1, 3),
        "annotation.human.action.task_description": ["do your thing!"],
    }

    time_start = time.time()
    action = policy_client.get_action(obs)
    print(f"Total time taken to get action from server: {time.time() - time_start} seconds")

    for key, value in action.items():
        print(f"Action: {key}: {value.shape}")
        import pdb; pdb.set_trace()
