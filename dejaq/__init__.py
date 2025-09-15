"""
.. include:: ../README.md
"""
__docformat__ = 'google'

import logging, sys
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(funcName)s: %(message)s", stream=sys.stdout)

from .queues import DejaQueue, PicklableDejaQueue
from .parallel import Parallel
from .remote import Actor, RemoteFunc, ActorDecorator

# Make imported classes appear as part of the top-level module for pdoc
for _cls in [DejaQueue, PicklableDejaQueue, Parallel, Actor, RemoteFunc, ActorDecorator]:
    _cls.__module__ = __name__