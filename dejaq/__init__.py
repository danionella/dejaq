"""
.. include:: ../README.md
"""
__docformat__ = 'google'

# Logging configuration is left to the application. dejaq logs to module loggers ("dejaq.*") and adds no
# handler, so when the application has configured none, warnings and errors (e.g. from worker processes)
# still reach stderr through logging's last-resort handler.

from .queues import LegacyDejaQueue, DejaQueue
from .parallel import Parallel
from .remote import Actor, RemoteFunc, ActorDecorator
from .stream import Source, MapNode

__all__ = ["DejaQueue", "Parallel", "Actor", "Source", "MapNode", "remote", "stream", "queues"]