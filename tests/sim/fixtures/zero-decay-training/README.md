# Historical zero-decay training regression

The numerical model helpers and retained acceptance data come from SNNLab commit `b361c3d254c22eaa1c2f5b7f4fec0572b3285094`. The retired benchmark runners are omitted; model equations, initialization, encoding and acceptance checks are preserved.

`test_snntorch_training_regression.py` uses these fixtures for 40 independently evolved clipped Adam updates and for rejecting an accuracy gap despite shared-checkpoint agreement. Keeping the fixture preserves numerical regression coverage after the obsolete benchmark directory is removed. The newer MNIST comparison has a different architecture and surrogate, so its results cannot replace this reference.
