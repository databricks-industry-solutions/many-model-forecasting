"""Pre-canned Databricks responses for Tier 1 mock tools.

Only contains responses for commands that cannot be executed on DuckDB:
- SHOW TABLES (DuckDB doesn't support SHOW TABLES IN catalog.schema)
- Non-SQL tool responses (workspace connect, warehouses, jobs)

All other SQL responses are now handled by the DuckDB backend.
"""

FIXTURES = {
    # SHOW TABLES: DuckDB doesn't support this Databricks-specific syntax
    "show_tables": (
        "database,tableName,isTemporary\n"
        "default,raw_timeseries,false\n"
        "default,raw_weekly_timeseries,false\n"
        "default,raw_monthly_timeseries,false\n"
        "default,metadata,false\n"
        "default,lookup_table,false"
    ),
    # Non-SQL tool responses
    "workspace_connected": '{"status": "connected", "workspace": "test-workspace"}',
    "warehouses": (
        "id,name,state,cluster_size,max_num_clusters,auto_stop_mins\nmock-wh-001,Mock Warehouse,RUNNING,Small,1,120"
    ),
    "create_job_success": '{"job_id": 12345}',
    "run_job_success": '{"run_id": 67890, "state": "TERMINATED", "result_state": "SUCCESS"}',
    "upload_notebook_success": '{"path": "/notebooks/uploaded"}',
}
