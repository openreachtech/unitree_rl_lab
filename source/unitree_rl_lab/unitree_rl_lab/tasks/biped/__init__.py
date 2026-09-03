"""Bipedal stances for quadrupeds: the robot walks on two legs, freeing the other two.

Sibling of ``locomotion`` and ``dynamic``, and organised the same way -- the skill and its MDP terms
live here, while the task that trains it as an expert for the mixture-of-experts policy is
registered under ``robots/go2`` and expressed on ``..multitask``'s shared observation layout. That
is the convention the acrobatics and locomotion experts already follow.
"""
