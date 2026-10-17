UTKFace seven-figure bundle

Open viewer.html in a browser. It is self-contained and needs no network connection. Use the seven page links, Plotly legend clicks/double-clicks and native zoom/pan tools. The full-data curves aggregate seeds 42/43/44. Other panels state their actual coverage. No new training or test-driven selection was performed.

Default moving means are five epochs, or ten for 200-epoch runs. Choose Raw to inspect unsmoothed curves. Bands are sample SD across optimization seeds, not confidence intervals. Plots show trained epochs only; epoch-zero stats remain in the data files. Default axes emphasize later training; Fit all data restores full training trajectories. Use native zoom/pan to inspect later training. Scalar label NLL is undefined. epoch-zero.json retains reconstruction provenance; historical training and checkpoint selections are unchanged.

Approved coverage adaptations: SD 1/3/5/10; eight available categorical 1024x200 settings; TailRL-versus-supervised and digit figures show full data only. No 1024x200 digit runs exist. The Statistical MaxRL figure uses five shared legend entries across its two estimator rows. Eta is noise RMS / signal SD; conventional SNR = 1/eta^2.

Each figure is supplied as PNG and vector PDF. all-seven.pdf contains all seven pages. bundle-data.json records source IDs, LR values, seeds, raw curves and validation-only selection decisions.
