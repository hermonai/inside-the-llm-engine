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

INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:55:56Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=f16,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=1,k=2048', 'perf', '1', '1', '', '407.101409', '61817088896.732086', '0.000000', '393304', '3974', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:55:58Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=f16,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=4,k=2048', 'perf', '1', '1', '', '1698.960765', '59249923893.586273', '0.000000', '393569', '994', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:56:00Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=f16,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=8,k=2048', 'perf', '1', '1', '', '3226.070423', '62406136764.301407', '0.000000', '393923', '497', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:56:02Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=f16,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=32,k=2048', 'perf', '1', '1', '', '6917.472000', '116416281554.880157', '0.000000', '396047', '250', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:56:03Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=f16,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=64,k=2048', 'perf', '1', '1', '', '7851.216931', '205141795228.724670', '0.000000', '398879', '189', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:56:05Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=f16,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=128,k=2048', 'perf', '1', '1', '', '8284.070312', '388845742549.942871', '0.000000', '404543', '128', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:56:06Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=f16,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=256,k=2048', 'perf', '1', '1', '', '8710.265625', '739638860783.881104', '0.000000', '415871', '128', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-09-24T10:56:08Z', 'd006858', 'MTL0', 'MUL_MAT_ID', 'type_a=f16,type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=512,k=2048', 'perf', '1', '1', '', '9870.634615', '1305377251825.052246', '0.000000', '438527', '104', '', '');
