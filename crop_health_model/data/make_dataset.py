import asyncio
import os
import zipfile

import aiohttp
import filetype
import rarfile
import httpx

from crop_health_model.data.metadata import all_datasets

harvard_dataverse_base_url = "https://dataverse.harvard.edu/api/access/datafile/"


async def download(url, file_path):

    folder, filename = os.path.split(file_path) 

    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    async with httpx.AsyncClient() as client:
        print(f"Downloading {url}")
        async with client.stream('GET', url, follow_redirects=True) as response:
            if response.status_code == 200:
                print(f"Download successful {filename}")
                with open(filename, 'wb') as f:
                    async for chunk in response.aiter_bytes(chunk_size=1024):
                        f.write(chunk)
                print(f"Download complete {filename}")
            else:
                print(f"Failed to download: {response.status_code}")
                return

async def download_all():

    download_info = [(harvard_dataverse_base_url + id, os.path.join(dataset["folder"], fname)) for dataset in all_datasets for id, fname in dataset["ids"]]
    download_info = download_info[:2]
    print(download_info)
    # Download and extract files concurrently
    tasks = [download(url, file_path) for url, file_path in download_info]
    await asyncio.gather(*tasks)

def delete_non_images(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                os.remove(os.path.join(root, file))

def extract(filename, folder):

    if filename.endswith(".zip"):
        with zipfile.ZipFile(filename, "r") as zip_ref:
            all_files = zip_ref.namelist()
            # remove files that are not images
            images = [file for file in all_files if file.lower().endswith('.jpg')]
            print(f"Extracting zip {filename}")
            zip_ref.extractall(folder, members=images)
            print(f"Extracted zip {filename}")
    elif filename.endswith(".rar"):
        with rarfile.RarFile(filename, "r") as rar_ref:
            all_files = rar_ref.namelist()
            # remove files that are not images
            images = [file for file in all_files if file.lower().endswith('.jpg')]
            print(f"Extracting rar {filename}")
            rar_ref.extractall(folder, members=images)
            print(f"Extracted rar {filename}")
    else:
        print("Unsupported file format.")

def extract_all():
    for dataset in all_datasets:
        for root, dirs, files in os.walk(dataset["folder"]):
            for file in files:
                extract(file, root)

if __name__ == "__main__":
    asyncio.run(download_all())
    delete_non_images('.data')
