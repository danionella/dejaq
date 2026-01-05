import pytest
from dejaq.remote import Actor, ActorDecorator

class Counter:
    def __init__(self, start=0):
        self.value = start
    def inc(self, n=1):
        self.value += n
        return self.value
    def get(self):
        return self.value

def test_actor_basic():
    actor = Actor(Counter, 10)
    assert actor.get() == 10
    assert actor.inc() == 11
    assert actor.inc(5) == 16
    assert actor.get() == 16
    actor.close()

def test_actor_context_manager():
    with Actor(Counter, 100) as actor:
        assert actor.inc() == 101
        assert actor.get() == 101

@ActorDecorator
class Greeter:
    def __init__(self, name):
        self.name = name
    def greet(self):
        return f"Hello, {self.name}!"

def test_actordecorator_basic():
    greeter = Greeter("Alice")
    assert greeter.greet() == "Hello, Alice!"
    greeter.close()