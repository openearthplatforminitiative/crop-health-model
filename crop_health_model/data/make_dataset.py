import asyncio
import os
import zipfile

import aiohttp
import filetype
import httpx
import rarfile
from tqdm import tqdm as sync_tqdm
from tqdm.asyncio import tqdm as async_tqdm

from crop_health_model.data.metadata import all_datasets

harvard_dataverse_base_url = "https://dataverse.harvard.edu/api/access/datafile/"


async def download(url, file_path):

    folder, filename = os.path.split(file_path)

    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Check if file already exists
    if os.path.exists(file_path):
        print(f"File already exists: {file_path}")
        return

    async with httpx.AsyncClient() as client:
        print(f"Downloading {url}")
        async with client.stream("GET", url, follow_redirects=True) as response:
            if response.status_code == 200:
                total_size = int(response.headers.get("Content-Length", 0))
                with open(file_path, "wb") as f:
                    with async_tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=filename,
                    ) as pbar:
                        async for chunk in response.aiter_bytes(chunk_size=1024):
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                print(f"Failed to download: {response.status_code}")
                return


async def download_all():

    download_info = [
        (
            harvard_dataverse_base_url + id,
            os.path.join(".data", dataset["folder"], fname),
        )
        for dataset in all_datasets
        for id, fname in dataset["ids"]
    ]
    # download_info = download_info[:3]
    print(download_info)
    # Download and extract files concurrently
    tasks = [download(url, file_path) for url, file_path in download_info]
    await asyncio.gather(*tasks)


def delete_non_images(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                os.remove(os.path.join(root, file))


def extract(file_path):
    folder, filename = os.path.split(file_path)
    # folder = os.path.dirname(file_path)
    if file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            all_files = zip_ref.namelist()
            # remove files that are not images
            images = [
                file for file in all_files if file.lower().endswith((".jpg", ".jpeg"))
            ]
            # print(f"Extracting zip {file_path}")
            # Use tqdm to create a progress bar
            for file in sync_tqdm(images, desc=f"Extracting {filename}", unit="file"):
                zip_ref.extract(member=file, path=folder)
            # print(f"Extracted zip {file_path}")
    elif file_path.endswith(".rar"):
        with rarfile.RarFile(file_path, "r") as rar_ref:
            all_files = rar_ref.namelist()
            # remove files that are not images
            images = [
                file for file in all_files if file.lower().endswith((".jpg", ".jpeg"))
            ]
            # print(f"Extracting rar {file_path}")
            # Use tqdm to create a progress bar
            for file in sync_tqdm(images, desc=f"Extracting {filename}", unit="file"):
                rar_ref.extract(member=file, path=folder)
            # print(f"Extracted rar {file_path}")
    else:
        print(f"Unsupported file format for {file_path}")


def extract_all():
    # print current directory
    print(f"current dir: {os.getcwd()}")
    for dataset in all_datasets:
        folder = os.path.join(".data", dataset["folder"])
        for file_name in os.listdir(folder):
            # check that file is a file:
            file_path = os.path.join(folder, file_name)
            if not os.path.isfile(file_path):
                continue
            # Some of the archives are multi-part RAR archives. So if the
            # file contains the substring ".part2.", ".part3.", etc. skip it.
            # But don't skip ".part1.", unarchiving it will unarchive the
            # other parts as well.
            if ".part" in file_name and not ".part1." in file_name:
                print(f"Skipping {file_name}")
                continue
            extract(file_path)

        # for root, dirs, files in os.walk(os.path.join(".data", dataset["folder"])):
        #     print(f"Extracting files in {root}")
        #     for file in files:
        #         # Some of the archives are multi-part RAR archives. So if the
        #         # file contains the substring ".part2.", ".part3.", etc. skip it.
        #         # But don't skip ".part1.", unarchiving it will unarchive the
        #         # other parts as well.
        #         if ".part" in file and not ".part1." in file:
        #             print(f"Skipping {file}")
        #             continue
        #         extract(file, root)


def delete_archives():
    for dataset in all_datasets:
        folder = os.path.join(".data", dataset["folder"])
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            if file_path.endswith((".zip", ".rar")):
                os.remove(file_path)


if __name__ == "__main__":
    asyncio.run(download_all())
    extract_all()
    # delete_non_images(".data")
