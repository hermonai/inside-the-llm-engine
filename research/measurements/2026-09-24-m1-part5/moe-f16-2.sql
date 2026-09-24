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

INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:56:48Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=f16,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=1,k=2048', 'perf', '1', '1', '', '408.224207', '61647064399.984467', '0.000000', '393304', '3974', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:56:50Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=f16,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=4,k=2048', 'perf', '1', '1', '', '1658.871227', '60681802384.351433', '0.000000', '393569', '994', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:56:52Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=f16,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=8,k=2048', 'perf', '1', '1', '', '3195.678068', '62999647552.151672', '0.000000', '393923', '497', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:56:54Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=f16,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=32,k=2048', 'perf', '1', '1', '', '6904.424000', '116636285372.972458', '0.000000', '396047', '250', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:56:55Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=f16,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=64,k=2048', 'perf', '1', '1', '', '8032.817460', '200504087632.578644', '0.000000', '398879', '126', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:56:56Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=f16,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=128,k=2048', 'perf', '1', '1', '', '8186.039062', '393502333351.466309', '0.000000', '404543', '128', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:56:57Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=f16,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=256,k=2048', 'perf', '1', '1', '', '9084.133929', '709198146422.874268', '0.000000', '415871', '112', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:56:59Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=f16,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=512,k=2048', 'perf', '1', '1', '', '9916.307692', '1299364873277.895020', '0.000000', '438527', '104', '', '');
