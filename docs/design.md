# Design & Theory

## Core Operator
Y = iFFT( D_theta ⊙ FFT(X W_v) ) W_o

- FFT/DCT is unitary/orthogonal ⇒ stable energy.
- D_theta per-head diagonal complex filter (gain + phase) over rFFT bins.
- Residual path cold-starts at identity (log_gain=0, phase=0).

## Stability
- rFFT/irFFT ensures real outputs.
- Spectral params initialized to identity; optional TV regularizer on phase.
- Token gate is rank-1 (small), disabled by default.

## Complexity
- O(n log n) over sequence length; broadcasts across heads/channels.
