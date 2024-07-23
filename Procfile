release: dvc remote add -d s3remote s3://tomsprojectbucket && dvc pull
web: uvicorn main:app --host 0.0.0.0 --port $PORT