Initial extraction from Pinglab working tree at eb77f161104155eae31b44986ddb3f45f1d33e24.

Includes the existing uncommitted simulator execution changes and the two new COBA-threshold/spiking-readout tests present at extraction. Changes in this extraction are packaging, imports, module entry points and the default training output path (now relative to the invoking working directory). Scientific schemas and numerical defaults are retained.

The pre-existing threshold test used an unsupported initial_voltage_mv field. Its fixture now uses an identical conductance pulse to place the two populations on opposite sides of their thresholds; simulator dynamics remain unchanged.
