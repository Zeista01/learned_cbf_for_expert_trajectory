# Clearance sweep — generalization, backstop ON

Reach = fraction of scenes where the tool arrived at the target.
Safety must remain 0 unsafe at every clearance.


## clearance 15 mm  (32 generalization scenes)

| method | reach | unsafe | min_sdf_mm | dev_max_mm | backstop_rate |
|---|---|---|---|---|---|
| ours_ta_cbf | 0.34 | 0/32 | 11.0 | 24.4 | 0.603 |
| b2_fixed_pose_cbf | 0.69 | 0/32 | 10.8 | 25.0 | 0.463 |
| b3a_oracle_sdf | 0.50 | 0/32 | 11.7 | 22.0 | 0.000 |
| b3b_cloud_esdf | 0.72 | 0/32 | 11.0 | 33.3 | 0.004 |
| b4_circle_cbf | 0.59 | 0/32 | 11.0 | 35.2 | 0.000 |
| b5_cncbf_pershape | 0.50 | 0/32 | 11.0 | 18.1 | 0.302 |

## clearance 20 mm  (24 generalization scenes)

| method | reach | unsafe | min_sdf_mm | dev_max_mm | backstop_rate |
|---|---|---|---|---|---|
| ours_ta_cbf | 0.54 | 0/24 | 11.0 | 21.1 | 0.448 |
| b2_fixed_pose_cbf | 0.67 | 0/24 | 11.0 | 22.0 | 0.387 |
| b3a_oracle_sdf | 0.46 | 0/24 | 11.7 | 20.4 | 0.000 |
| b3b_cloud_esdf | 0.71 | 0/24 | 11.0 | 30.9 | 0.006 |
| b4_circle_cbf | 0.71 | 0/24 | 11.0 | 33.8 | 0.000 |
| b5_cncbf_pershape | 0.50 | 0/24 | 11.0 | 17.0 | 0.185 |

## clearance 25 mm  (24 generalization scenes)

| method | reach | unsafe | min_sdf_mm | dev_max_mm | backstop_rate |
|---|---|---|---|---|---|
| ours_ta_cbf | 0.62 | 0/24 | 11.0 | 22.4 | 0.377 |
| b2_fixed_pose_cbf | 0.67 | 0/24 | 11.0 | 18.6 | 0.441 |
| b3a_oracle_sdf | 0.46 | 0/24 | 11.7 | 20.7 | 0.000 |
| b3b_cloud_esdf | 0.79 | 0/24 | 11.0 | 32.4 | 0.004 |
| b4_circle_cbf | 0.79 | 0/24 | 11.0 | 34.3 | 0.006 |
| b5_cncbf_pershape | 0.54 | 0/24 | 11.0 | 17.1 | 0.225 |

## clearance 30 mm  (24 generalization scenes)

| method | reach | unsafe | min_sdf_mm | dev_max_mm | backstop_rate |
|---|---|---|---|---|---|
| ours_ta_cbf | 0.50 | 0/24 | 11.0 | 19.4 | 0.464 |
| b2_fixed_pose_cbf | 0.62 | 0/24 | 11.0 | 18.4 | 0.356 |
| b3a_oracle_sdf | 0.71 | 0/24 | 11.7 | 20.4 | 0.000 |
| b3b_cloud_esdf | 0.88 | 0/24 | 11.0 | 30.0 | 0.003 |
| b4_circle_cbf | 0.83 | 0/24 | 11.0 | 32.5 | 0.000 |
| b5_cncbf_pershape | 0.62 | 0/24 | 11.0 | 18.6 | 0.172 |

See README.md for the interpretation and the backstop-off comparison.
