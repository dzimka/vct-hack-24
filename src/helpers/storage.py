import os


def get_storage_options():
    bucket = os.environ.get("AWS_S3_BUCKET")
    region = os.environ.get("AWS_REGION")
    key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if bucket:
        return {
            "bucket": bucket,
            "AWS_REGION": region,
            "AWS_ACCESS_KEY_ID": key_id,
            "AWS_SECRET_ACCESS_KEY": access_key,
        }
    return {}
