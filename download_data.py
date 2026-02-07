import requests, os
import pandas as pd
from pathlib import Path

def process_dataset(csv_path):
    csv_path = Path(csv_path)
    dataset_name = csv_path.stem
    output_dir = csv_path.parent / dataset_name
    output_dir.mkdir(exist_ok=True)

    df = pd.read_csv(csv_path, header=None)

    df = df.rename(columns={
        0: "WorkerID",
        1: "TaskID",
        2: "WorkerLabel",
        3: "GoldLabel"
    })

    worker_map = {w: i for i, w in enumerate(df["WorkerID"].astype(str).unique())}
    item_map = {t: i for i, t in enumerate(df["TaskID"].astype(str).unique())}

    df["worker"] = df["WorkerID"].astype(str).map(worker_map)
    df["item"] = df["TaskID"].astype(str).map(item_map)

    label_df = df[["item", "worker", "WorkerLabel"]] \
        .rename(columns={"WorkerLabel": "label"})
    label_df.to_csv(output_dir / "label.csv", index=False)

    truth_df = df[["item", "GoldLabel"]] \
        .drop_duplicates(subset=["item"]) \
        .rename(columns={"GoldLabel": "truth"})
    truth_df.to_csv(output_dir / "truth.csv", index=False)





if __name__ == "__main__":

    owner = "orchidproject"
    repo = "active-crowd-toolkit"
    path = "Data"
    branch = "master"

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    headers = {"Accept": "application/vnd.github.v3+json"}

    files = requests.get(url, headers=headers).json()
    os.makedirs(path, exist_ok=True)

    for file in files:
        if file["type"] == "file":
            r = requests.get(file["download_url"])
            with open(os.path.join(path, file["name"]), "wb") as f:
                f.write(r.content)
    os.rename("Data", "data")
    for item in os.listdir("data"):
        if item.endswith(".csv"):
            csv_path = os.path.join("data", item)
            process_dataset(csv_path)
            os.remove(csv_path)

