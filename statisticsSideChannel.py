from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import uuid

class StatisticalSideChannel(SideChannel):
    def __init__(self):
        super().__init__(uuid.UUID("ad041eb0-4a51-451f-a363-bb3a1a5d47bc"))
        self.agent_win_count = 0
        self.opponent_win_count = 0

    def on_message_received(self, msg: IncomingMessage):
        msg_list = msg.read_float32_list()
        self.agent_win_count = msg_list[0]
        self.opponent_win_count = msg_list[1]

    def send_string(self, data: str):
        msg = OutgoingMessage()
        msg.write_string(data)
        super().queue_message_to_send(msg)