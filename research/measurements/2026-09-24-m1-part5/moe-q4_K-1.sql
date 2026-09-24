CREATE TABLE IF NOT EXISTS test_backend_ops (
  test_time TEXT,
  build_commit TEXT,
  backend_name TEXT,
  op_name TEXT,
  op_params TEXT,
  test_mode TEXT,
  supported INTEGER,
  passed INTEGER,
  error_message TEXT,
  time_us REAL,
  flops REAL,
  bandwidth_gb_s REAL,
  memory_kb INTEGER,
  n_runs INTEGER,
  device_description TEXT,
  backend_reg_name TEXT
);

INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:55:19Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=1,k=2048', 'perf', '1', '1', '', '102.181178', '246286298294.866852', '0.000000', '110680', '11922', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:55:25Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=4,k=2048', 'perf', '1', '1', '', '463.041583', '217395801441.203949', '0.000000', '110945', '2982', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:55:30Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=8,k=2048', 'perf', '1', '1', '', '924.254192', '217825998283.097351', '0.000000', '111299', '1491', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:55:34Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=32,k=2048', 'perf', '1', '1', '', '6362.472000', '126571302474.887115', '0.000000', '113423', '250', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:55:38Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=64,k=2048', 'perf', '1', '1', '', '7043.380952', '228670399469.951508', '0.000000', '116255', '189', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:55:44Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=128,k=2048', 'perf', '1', '1', '', '7208.018750', '446894713197.020996', '0.000000', '121919', '160', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:55:48Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=256,k=2048', 'perf', '1', '1', '', '7321.805556', '879899212716.960327', '0.000000', '133247', '144', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:55:54Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=512,k=2048', 'perf', '1', '1', '', '10596.479167', '1215960668193.641113', '0.000000', '155903', '96', '', '');
