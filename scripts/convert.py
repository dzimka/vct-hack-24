import os
import pathlib

import polars as pl

FOLDER = "game-changers/esports-data"
INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/parquet"


def convert_esports_data(league: str = "game-changers"):
    input_directory = f"{INPUT_DIR}/{league}/esports-data"
    output_directory = f"{OUTPUT_DIR}/{league}/esports-data"

    esports_data_files = ["leagues", "tournaments", "players", "teams", "mapping_data"]

    for obj_name in esports_data_files:
        # read JSON
        df = pl.read_json(f"{input_directory}/{obj_name}.json")

        # write to Parquet
        parquet_path: pathlib.Path = f"{output_directory}/{obj_name}"

        if not os.path.exists(parquet_path):
            os.makedirs(parquet_path)

        df.write_parquet(parquet_path + "/" + "data.parquet")


convert_esports_data()
