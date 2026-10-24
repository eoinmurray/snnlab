# snnlang report — scikit_digits_coba

Populations: 1 (64 units)
Projections: 1
Operations: 2
Parameters: 2 tensors / 4,736 scalars
Estimated state: 64 scalars per sample and timestep
Estimated dense projection edges: 4,096
Trainable this recipe: 2 tensors
Outputs: class_scores
Recurrent paths: none
Diagnostics: 0 errors, 0 warnings

## Populations
- hidden: 64 × coba_lif (spiking)

## Projections
- pixels_to_hidden: pixels.value → hidden.excitatory [feedforward, excitatory]

## Parameters
- classifier.weight: [10, 64] 1 (selected)
- pixels_to_hidden.weight: [64, 64] uS (selected)
