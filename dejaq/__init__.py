"""
.. include:: ../README.md
"""
__docformat__ = 'google'

import logging, sys
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(funcName)s: %(message)s", stream=sys.stdout)

from .queues import LegacyDejaQueue, DejaQueue
from .parallel import Parallel
from .remote import Actor, RemoteFunc, ActorDecorator