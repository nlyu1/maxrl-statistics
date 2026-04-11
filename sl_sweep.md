# SL Sweep

This report tracks two-epoch validation correlation sweeps over train_lr, head_lr, and weight_decay with 100-step linear warmup.

Results root: `/tmp/sl_sweep/results`

## Best By Epoch 1

| rank | snr | train_lr | head_lr | weight_decay | epoch1_val_corr | epoch2_val_corr | delta_e2_e1 | epoch1_val_rsq | run_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 3.00e-05 | 2.00e-04 | 0.00e+00 | 0.994957 | 0.995082 | 0.000125 | 0.927126 | corr_snr1_coarse__snr1__tlr3em05__hlr2em04__wd0__seed0 |
| 2 | 1 | 3.00e-05 | 2.00e-04 | 5.00e-02 | 0.994908 | 0.994859 | -0.000049 | 0.886556 | corr_snr1_coarse__snr1__tlr3em05__hlr2em04__wd0p05__seed0 |
| 3 | 1 | 1.00e-04 | 5.00e-04 | 5.00e-02 | 0.994901 | 0.995206 | 0.000305 | 0.882628 | corr_snr1_coarse__snr1__tlr1em04__hlr5em04__wd0p05__seed0 |
| 4 | 1 | 1.00e-04 | 2.00e-04 | 0.00e+00 | 0.994898 | 0.994697 | -0.000201 | 0.924444 | corr_snr1_coarse__snr1__tlr1em04__hlr2em04__wd0__seed0 |
| 5 | 1 | 3.00e-05 | 1.00e-03 | 0.00e+00 | 0.994734 | 0.995178 | 0.000443 | 0.939952 | corr_snr1_coarse__snr1__tlr3em05__hlr1em03__wd0__seed0 |
| 6 | 0.7 | 3.00e-05 | 5.00e-04 | 0.00e+00 | 0.994698 | 0.994964 | 0.000265 | 0.988812 | corr_snr07_transfer__snr0p7__tlr3em05__hlr5em04__wd0__seed0 |
| 7 | 1 | 1.00e-04 | 1.00e-03 | 0.00e+00 | 0.994533 | 0.993846 | -0.000687 | 0.981331 | corr_snr1_coarse__snr1__tlr1em04__hlr1em03__wd0__seed0 |
| 8 | 1 | 3.00e-05 | 5.00e-04 | 0.00e+00 | 0.994431 | 0.994760 | 0.000329 | 0.973953 | corr_snr1_coarse__snr1__tlr3em05__hlr5em04__wd0__seed0 |
| 9 | 1 | 3.00e-05 | 5.00e-04 | 5.00e-02 | 0.994342 | 0.994470 | 0.000128 | 0.945316 | corr_snr1_coarse__snr1__tlr3em05__hlr5em04__wd0p05__seed0 |
| 10 | 0.7 | 3.00e-05 | 2.00e-04 | 5.00e-02 | 0.994273 | 0.994625 | 0.000351 | 0.982418 | corr_snr07_transfer__snr0p7__tlr3em05__hlr2em04__wd0p05__seed0 |
| 11 | 1 | 1.00e-04 | 2.00e-04 | 5.00e-02 | 0.994173 | 0.993378 | -0.000794 | 0.967350 | corr_snr1_coarse__snr1__tlr1em04__hlr2em04__wd0p05__seed0 |
| 12 | 0.7 | 3.00e-05 | 2.00e-04 | 0.00e+00 | 0.993881 | 0.995400 | 0.001520 | 0.972214 | corr_snr07_transfer__snr0p7__tlr3em05__hlr2em04__wd0__seed0 |
| 13 | 1 | 3.00e-05 | 1.00e-03 | 5.00e-02 | 0.993750 | 0.995166 | 0.001416 | 0.944419 | corr_snr1_coarse__snr1__tlr3em05__hlr1em03__wd0p05__seed0 |
| 14 | 1 | 1.00e-04 | 1.00e-03 | 5.00e-02 | 0.992850 | 0.993411 | 0.000561 | 0.977951 | corr_snr1_coarse__snr1__tlr1em04__hlr1em03__wd0p05__seed0 |
| 15 | 0.7 | 3.00e-05 | 5.00e-04 | 5.00e-02 | 0.992818 | 0.995145 | 0.002327 | 0.983440 | corr_snr07_transfer__snr0p7__tlr3em05__hlr5em04__wd0p05__seed0 |
| 16 | 1 | 1.00e-04 | 5.00e-04 | 0.00e+00 | 0.992785 | 0.993847 | 0.001062 | 0.971597 | corr_snr1_coarse__snr1__tlr1em04__hlr5em04__wd0__seed0 |
| 17 | 0.5 | 3.00e-05 | 5.00e-04 | 0.00e+00 | 0.992318 | 0.993654 | 0.001336 | 0.960072 | corr_snr_transfer3__snr0p5__snr07_head_fast__tlr3em05__hlr5em04__wd0__seed0 |
| 18 | 0.5 | 3.00e-05 | 2.00e-04 | 0.00e+00 | 0.992066 | 0.994002 | 0.001936 | 0.977885 | corr_snr_transfer3__snr0p5__snr1_best__tlr3em05__hlr2em04__wd0__seed0 |
| 19 | 1 | 1.00e-05 | 2.00e-04 | 0.00e+00 | 0.988711 | 0.990923 | 0.002211 | 0.936579 | corr_snr1_coarse__snr1__tlr1em05__hlr2em04__wd0__seed0 |
| 20 | 1 | 1.00e-05 | 1.00e-03 | 5.00e-02 | 0.988390 | 0.990673 | 0.002283 | 0.895757 | corr_snr1_coarse__snr1__tlr1em05__hlr1em03__wd0p05__seed0 |
| 21 | 1 | 1.00e-05 | 5.00e-04 | 5.00e-02 | 0.988336 | 0.991015 | 0.002679 | 0.955769 | corr_snr1_coarse__snr1__tlr1em05__hlr5em04__wd0p05__seed0 |
| 22 | 1 | 1.00e-05 | 5.00e-04 | 0.00e+00 | 0.987844 | 0.989719 | 0.001874 | 0.967843 | corr_snr1_coarse__snr1__tlr1em05__hlr5em04__wd0__seed0 |
| 23 | 1 | 1.00e-05 | 1.00e-03 | 0.00e+00 | 0.987627 | 0.991254 | 0.003627 | 0.972558 | corr_snr1_coarse__snr1__tlr1em05__hlr1em03__wd0__seed0 |
| 24 | 1 | 1.00e-05 | 2.00e-04 | 5.00e-02 | 0.987470 | 0.990170 | 0.002701 | 0.971808 | corr_snr1_coarse__snr1__tlr1em05__hlr2em04__wd0p05__seed0 |


## Non-Completed Runs

_No failed, running, or incomplete run records._


## SNR 1

Best epoch-1 config by validation correlation: `train_lr=3.000e-05`, `head_lr=2.000e-04`, `weight_decay=0.000e+00`.

| rank | snr | train_lr | head_lr | weight_decay | epoch1_val_corr | epoch2_val_corr | delta_e2_e1 | epoch1_val_rsq | run_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 3.00e-05 | 2.00e-04 | 0.00e+00 | 0.994957 | 0.995082 | 0.000125 | 0.927126 | corr_snr1_coarse__snr1__tlr3em05__hlr2em04__wd0__seed0 |
| 2 | 1 | 3.00e-05 | 2.00e-04 | 5.00e-02 | 0.994908 | 0.994859 | -0.000049 | 0.886556 | corr_snr1_coarse__snr1__tlr3em05__hlr2em04__wd0p05__seed0 |
| 3 | 1 | 1.00e-04 | 5.00e-04 | 5.00e-02 | 0.994901 | 0.995206 | 0.000305 | 0.882628 | corr_snr1_coarse__snr1__tlr1em04__hlr5em04__wd0p05__seed0 |
| 4 | 1 | 1.00e-04 | 2.00e-04 | 0.00e+00 | 0.994898 | 0.994697 | -0.000201 | 0.924444 | corr_snr1_coarse__snr1__tlr1em04__hlr2em04__wd0__seed0 |
| 5 | 1 | 3.00e-05 | 1.00e-03 | 0.00e+00 | 0.994734 | 0.995178 | 0.000443 | 0.939952 | corr_snr1_coarse__snr1__tlr3em05__hlr1em03__wd0__seed0 |
| 6 | 1 | 1.00e-04 | 1.00e-03 | 0.00e+00 | 0.994533 | 0.993846 | -0.000687 | 0.981331 | corr_snr1_coarse__snr1__tlr1em04__hlr1em03__wd0__seed0 |
| 7 | 1 | 3.00e-05 | 5.00e-04 | 0.00e+00 | 0.994431 | 0.994760 | 0.000329 | 0.973953 | corr_snr1_coarse__snr1__tlr3em05__hlr5em04__wd0__seed0 |
| 8 | 1 | 3.00e-05 | 5.00e-04 | 5.00e-02 | 0.994342 | 0.994470 | 0.000128 | 0.945316 | corr_snr1_coarse__snr1__tlr3em05__hlr5em04__wd0p05__seed0 |
| 9 | 1 | 1.00e-04 | 2.00e-04 | 5.00e-02 | 0.994173 | 0.993378 | -0.000794 | 0.967350 | corr_snr1_coarse__snr1__tlr1em04__hlr2em04__wd0p05__seed0 |
| 10 | 1 | 3.00e-05 | 1.00e-03 | 5.00e-02 | 0.993750 | 0.995166 | 0.001416 | 0.944419 | corr_snr1_coarse__snr1__tlr3em05__hlr1em03__wd0p05__seed0 |
| 11 | 1 | 1.00e-04 | 1.00e-03 | 5.00e-02 | 0.992850 | 0.993411 | 0.000561 | 0.977951 | corr_snr1_coarse__snr1__tlr1em04__hlr1em03__wd0p05__seed0 |
| 12 | 1 | 1.00e-04 | 5.00e-04 | 0.00e+00 | 0.992785 | 0.993847 | 0.001062 | 0.971597 | corr_snr1_coarse__snr1__tlr1em04__hlr5em04__wd0__seed0 |
| 13 | 1 | 1.00e-05 | 2.00e-04 | 0.00e+00 | 0.988711 | 0.990923 | 0.002211 | 0.936579 | corr_snr1_coarse__snr1__tlr1em05__hlr2em04__wd0__seed0 |
| 14 | 1 | 1.00e-05 | 1.00e-03 | 5.00e-02 | 0.988390 | 0.990673 | 0.002283 | 0.895757 | corr_snr1_coarse__snr1__tlr1em05__hlr1em03__wd0p05__seed0 |
| 15 | 1 | 1.00e-05 | 5.00e-04 | 5.00e-02 | 0.988336 | 0.991015 | 0.002679 | 0.955769 | corr_snr1_coarse__snr1__tlr1em05__hlr5em04__wd0p05__seed0 |
| 16 | 1 | 1.00e-05 | 5.00e-04 | 0.00e+00 | 0.987844 | 0.989719 | 0.001874 | 0.967843 | corr_snr1_coarse__snr1__tlr1em05__hlr5em04__wd0__seed0 |
| 17 | 1 | 1.00e-05 | 1.00e-03 | 0.00e+00 | 0.987627 | 0.991254 | 0.003627 | 0.972558 | corr_snr1_coarse__snr1__tlr1em05__hlr1em03__wd0__seed0 |
| 18 | 1 | 1.00e-05 | 2.00e-04 | 5.00e-02 | 0.987470 | 0.990170 | 0.002701 | 0.971808 | corr_snr1_coarse__snr1__tlr1em05__hlr2em04__wd0p05__seed0 |


## SNR 0.7

Best epoch-1 config by validation correlation: `train_lr=3.000e-05`, `head_lr=5.000e-04`, `weight_decay=0.000e+00`.

| rank | snr | train_lr | head_lr | weight_decay | epoch1_val_corr | epoch2_val_corr | delta_e2_e1 | epoch1_val_rsq | run_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.7 | 3.00e-05 | 5.00e-04 | 0.00e+00 | 0.994698 | 0.994964 | 0.000265 | 0.988812 | corr_snr07_transfer__snr0p7__tlr3em05__hlr5em04__wd0__seed0 |
| 2 | 0.7 | 3.00e-05 | 2.00e-04 | 5.00e-02 | 0.994273 | 0.994625 | 0.000351 | 0.982418 | corr_snr07_transfer__snr0p7__tlr3em05__hlr2em04__wd0p05__seed0 |
| 3 | 0.7 | 3.00e-05 | 2.00e-04 | 0.00e+00 | 0.993881 | 0.995400 | 0.001520 | 0.972214 | corr_snr07_transfer__snr0p7__tlr3em05__hlr2em04__wd0__seed0 |
| 4 | 0.7 | 3.00e-05 | 5.00e-04 | 5.00e-02 | 0.992818 | 0.995145 | 0.002327 | 0.983440 | corr_snr07_transfer__snr0p7__tlr3em05__hlr5em04__wd0p05__seed0 |


## SNR 0.5

Best epoch-1 config by validation correlation: `train_lr=3.000e-05`, `head_lr=5.000e-04`, `weight_decay=0.000e+00`.

| rank | snr | train_lr | head_lr | weight_decay | epoch1_val_corr | epoch2_val_corr | delta_e2_e1 | epoch1_val_rsq | run_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.5 | 3.00e-05 | 5.00e-04 | 0.00e+00 | 0.992318 | 0.993654 | 0.001336 | 0.960072 | corr_snr_transfer3__snr0p5__snr07_head_fast__tlr3em05__hlr5em04__wd0__seed0 |
| 2 | 0.5 | 3.00e-05 | 2.00e-04 | 0.00e+00 | 0.992066 | 0.994002 | 0.001936 | 0.977885 | corr_snr_transfer3__snr0p5__snr1_best__tlr3em05__hlr2em04__wd0__seed0 |

