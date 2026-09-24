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

INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:56:13Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=1,k=2048', 'perf', '1', '1', '', '102.478192', '245572483397.108215', '0.000000', '110680', '11922', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:56:18Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=4,k=2048', 'perf', '1', '1', '', '468.929577', '214666126509.280975', '0.000000', '110945', '2982', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:56:22Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=8,k=2048', 'perf', '1', '1', '', '917.207914', '219499405635.011932', '0.000000', '111299', '1491', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:56:26Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=32,k=2048', 'perf', '1', '1', '', '6190.256000', '130092579046.811646', '0.000000', '113423', '250', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:56:32Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=64,k=2048', 'perf', '1', '1', '', '7092.984127', '227071244932.394653', '0.000000', '116255', '189', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:56:38Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=128,k=2048', 'perf', '1', '1', '', '7204.756250', '447097078683.265686', '0.000000', '121919', '160', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:56:42Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=256,k=2048', 'perf', '1', '1', '', '7313.979167', '880840756747.210693', '0.000000', '133247', '144', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:56:45Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=512,k=2048', 'perf', '1', '1', '', '10594.000000', '1216245222578.818359', '0.000000', '155903', '96', '', '');
