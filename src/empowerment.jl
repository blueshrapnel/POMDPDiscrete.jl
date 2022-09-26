"""
Empowerment

Empowerment measures the perceived potential for an agent to influence the environment.  Thie can be achieved either by moving, or changing the state of the agent, or by changing the environment.
Maximising empowerment can be used as intrinsic motivation for an agent's behaviour.

[All else being equal be empowered](https://uhra.herts.ac.uk/bitstream/handle/2299/1918/901933.pdf)

n-step empowerment, view the sequence of actions ``A_t^n`` as the transmitted signal and ``S_{t+n}`` as the received signal.  The system's dynamics induce a conditional probability distribution ``p(s_{t+n}|A_t^n)`` between the sequence of actions and the state of the sensor after ``n`` time steps.  This conditional distribution characterises the communication channel.

Empowerment is measured in bits, it is zero when the agent has no control over what it is sensing.
"""
