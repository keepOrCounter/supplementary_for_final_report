# [Resistance port] GRAIL's team enum, lifted out of its agent_base.py so the
# vendored files below can import it without the rest of that module, which
# pulls in the OpenAI client and their websocket protocol.
from enum import Enum


class ATEAM(Enum):
    GOOD = 1
    EVIL = 2
