call columnstore_info.table_usage('columnstore_bts', NULL);
SELECT table_name, (data_length + index_length) / (1024 * 1024) "Size in MB"  FROM information_schema.tables WHERE table_schema = 'innodb_bts';
