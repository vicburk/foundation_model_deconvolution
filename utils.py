import pandas as pd
import numpy as np
import os
import time
import requests
import itertools
import scanpy as sc
import math
from geneformer import TranscriptomeTokenizer, EmbExtractor
import mygene

def download_file(article_id, file_id, folder="downloads"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    base_url = "https://api.figshare.com/v2"
    url = f"{base_url}/articles/{article_id}/files/{file_id}"

    try:
        # Get metadata
        r = requests.get(url)
        r.raise_for_status()
        metadata = r.json()

        filename = metadata["name"]
        download_url = metadata["download_url"]
        filesize = metadata["size"]

        filepath = os.path.join(folder, filename)
        print(f"Downloading: {filename}")
        print(f"File size: {filesize / (1024 * 1024):.2f} MB\n")

        # Start download
        with requests.get(download_url, stream=True) as r2:
            r2.raise_for_status()
            total = int(r2.headers.get("content-length", filesize))
            downloaded = 0
            start_time = time.time()

            with open(filepath, "wb") as f:
                for chunk in r2.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Progress calculations
                        percent = downloaded / total * 100
                        elapsed = time.time() - start_time
                        speed = downloaded / elapsed / (1024 * 1024)  # MB/s

                        print(
                            f"\r{percent:6.2f}% "
                            f"{downloaded / (1024 * 1024):8.2f}MB "
                            f"at {speed:6.2f} MB/s",
                            end="",
                        )

        print(f"\n\nDownload complete: {filepath}")

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")


def load_and_clean_adata(file, path):
  adata = sc.read_h5ad(os.path.join(path, file))
  gene_symbols = list(adata.var.index)
  mg = mygene.MyGeneInfo()

  # Query for gene symbols, requesting ensembl.gene field
  results = mg.querymany(gene_symbols, scopes='symbol', fields='ensembl.gene', species='human', returnall=True, verbose=False)

  xx_data = []
  for res in results['out']:
      hgnc_symbol = res.get('query')
      ensembl_id = None

      # Handle cases where 'ensembl' might be a list of dicts or a single dict
      if 'ensembl' in res and res['ensembl'] is not None:
          ensembl_entries = res['ensembl']
          if isinstance(ensembl_entries, list):
              for item in ensembl_entries:
                  if isinstance(item, dict) and 'gene' in item:
                      ensembl_id = item['gene']
                      break # Take the first Ensembl gene ID found
          elif isinstance(ensembl_entries, dict) and 'gene' in ensembl_entries:
              ensembl_id = ensembl_entries['gene']

      if hgnc_symbol and ensembl_id: # Only add if both symbol and Ensembl ID are found
          xx_data.append({'hgnc_symbol': hgnc_symbol, 'ensembl_gene_id': ensembl_id})

  xx = pd.DataFrame(xx_data)

  # The subsequent cell expects 'xx' to be a DataFrame. If no matches were found, xx might be empty.
  # Ensure 'ensembl_gene_id' and 'hgnc_symbol' columns exist even if empty for consistency.
  if 'ensembl_gene_id' not in xx.columns:
      xx['ensembl_gene_id'] = pd.Series(dtype=str)
  if 'hgnc_symbol' not in xx.columns:
      xx['hgnc_symbol'] = pd.Series(dtype=str)

  print("Gene symbols mapped to Ensembl IDs successfully using mygene.")
  print(f"Found {len(xx)} mappings.")
  xx_dict = dict(zip(xx["hgnc_symbol"], xx["ensembl_gene_id"]))
  adata.var['ensembl_id'] = [xx_dict.get(i, "None") for i in gene_symbols]
  adata = adata[:, adata.var['ensembl_id'] != 'None']
  adata_raw = adata
  adata_raw.X = adata.layers['counts'].round(0).astype(int)
  adata_raw = adata_raw[adata_raw.obs['study'] != 'Villani', :]
  adata_raw.obs['n_counts'] = adata_raw.X.sum(axis=1)
  return adata_raw