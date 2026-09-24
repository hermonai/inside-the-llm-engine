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

INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:57:05Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=1,k=2048', 'perf', '1', '1', '', '103.145697', '243983265656.829346', '0.000000', '110680', '11922', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:57:11Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=4,k=2048', 'perf', '1', '1', '', '459.407780', '219115348889.086792', '0.000000', '110945', '2982', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:57:15Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=8,k=2048', 'perf', '1', '1', '', '917.183769', '219505183958.418182', '0.000000', '111299', '1491', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:57:19Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=32,k=2048', 'perf', '1', '1', '', '6184.160000', '130220817055.186142', '0.000000', '113423', '250', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:57:23Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=64,k=2048', 'perf', '1', '1', '', '7040.402116', '228767151275.029388', '0.000000', '116255', '189', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:57:27Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=128,k=2048', 'perf', '1', '1', '', '7205.981250', '447021073223.025635', '0.000000', '121919', '160', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:57:31Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=256,k=2048', 'perf', '1', '1', '', '7316.631944', '880521391935.231201', '0.000000', '133247', '144', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:57:37Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=q4_K,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=512,k=2048', 'perf', '1', '1', '', '10888.645833', '1183333729944.227051', '0.000000', '155903', '96', '', '');
